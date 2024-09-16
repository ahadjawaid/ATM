from atm.dataloader.utils import ImgViewDiffTranslationAug
import torchvision
from tqdm.auto import tqdm
import h5py
from typing import Sequence, Tuple, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from atm.dataloader.track_dataloader import ATMPretrainDataset
from atm.dataloader.bc_dataloader import BCDataset
from typing import Mapping, Tuple, List
from atm.dataloader.base_dataset import convert_points_to_tracks

class CachedAugmentedDataset(Dataset):
    def __init__(self, *args, dataset: str = 'atm', **kwargs):
        super().__init__()
        dataset_path = Path(kwargs['dataset_dir'][0]).parent.parent/f"{dataset}_train_dataset.hdf5"
        self.file = h5py.File(dataset_path, "r")
        self.data = self.file['data']
        self.length = len(self.data)

        self.use_points = kwargs.get('use_points')
    
        self.keys = ['vids', 'tracks', 'vis', 'task_emb', 'intrinsic', 'depth'] if dataset == 'atm' else \
                    ['obs', 'track_transformer_obs', 'tracks', 'task_embs', 'actions', 'extra_states', 'depth', 'intrinsics']

    def __getitem__(self, index):
        items = get_data(self.data[str(index)])
        items = to_tensor(items)

        if not self.use_points:
            intrinsic_key = 'intrinsic' if 'intrinsic' in items else 'intrinsics'
            tracks, depth, intrinsic = items['tracks'], items['depth'], items[intrinsic_key]
            if intrinsic.shape[0] == 2:
                track_views = [convert_points_to_tracks(tracks[i], intrinsic[i], depth.shape[-2:]) for i in range(intrinsic.shape[0])]
                items['tracks'] = torch.stack(track_views, dim=0)
            else:
                items['tracks'] = convert_points_to_tracks(tracks, intrinsic, depth.shape[-2:])

        items = [items[key] for key in self.keys]
        return tuple(items)

    def __len__(self) -> int:
        return len(self.data)
    
    def __del__(self):
        self.file.close()

def create_cached_augmented_dataset(*args, dataset_name: str = 'atm', **kwargs):
    dataset = ATMPretrainDataset(*args, **kwargs) if dataset_name == 'atm' else BCDataset(*args, **kwargs)
    dataset_dir = Path(dataset.dataset_dir[0]).parent.parent
    dataset_len = len(dataset)
    img_size = dataset.img_size
    aug_prob = dataset.aug_prob
    augment_track = dataset.augment_track

    keys = ['vids', 'tracks', 'vis', 'task_emb', 'intrinsic', 'depth'] if dataset_name == 'atm' else \
           ['obs', 'track_transformer_obs', 'tracks', 'task_embs', 'actions', 'extra_states', 'depth', 'intrinsics']

    file = h5py.File(dataset_dir/f"{dataset_name}_train_dataset.hdf5", "a")
    if 'data' not in file:
        file.create_group('data')
    
    if len(file['data']) < dataset_len:
        indices = range(len(file['data']), dataset_len)
        for idx in tqdm(indices, desc="Caching dataset"):
            values = dataset.__getitem__(idx)
            data = dict(zip(keys, values))

            current_idx = len(file['data'])
            group = file.create_group(f'data/{current_idx}')
            create_dataset(group, **data)

    del dataset

    color_augmentor = torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
    translation_augmentor = ImgViewDiffTranslationAug(input_shape=img_size, translation=8, augment_track=augment_track)
    
    
    n_augmentations = int((dataset_len * (1 / (1 - aug_prob))) - dataset_len) - (len(file['data']) - dataset_len)
    augmentation_indices = np.random.choice(dataset_len, n_augmentations)
    for idx in tqdm(augmentation_indices, desc="Augmenting dataset"):
        group = file[f'data/{idx}']
        data = get_data(group)

        obs_key = 'vids' if 'vids' in data else 'obs'

        vids, depth, tracks = data[obs_key], data['depth'], data['tracks']
        
        len_vids_shape, len_depth_shape, len_tracks_shape = len(vids.shape), len(depth.shape), len(tracks.shape)
        
        if len_vids_shape == 4:
            vids = vids[None, None]

        if len_depth_shape == 4:
            depth = depth[None, None]
        
        if len_tracks_shape == 3:
            tracks = tracks[None, None]

        if len_vids_shape == 5:
            vids = np.expand_dims(vids, axis=2)

        vids, depth, tracks = map(lambda x: torch.from_numpy(x) if not isinstance(x, torch.Tensor) else x, 
                                  (vids, depth, tracks))
        
        vid_views, track_views, depth_views = [], [], []
        for view in range(vids.shape[0]):
            current_vids, current_tracks, current_depth = vids[view], tracks[view], depth[view]
            current_tracks = current_tracks[None, None, ...]
            current_vids = color_augmentor.forward(current_vids / 255.)
            current_vids, current_tracks, current_depth = translation_augmentor.forward((current_vids, current_tracks, current_depth))
            current_vids = current_vids * 255.
            
            current_tracks = current_tracks[0, 0, ...]

            vid_views.append(current_vids)
            track_views.append(current_tracks)
            depth_views.append(current_depth)
            
        vids = np.stack(vid_views, axis=0)
        tracks = np.stack(track_views, axis=0)
        depth = np.stack(depth_views, axis=0)

        if len_vids_shape == 4:
            vids = vids[0, ...]
        
        if len_depth_shape == 4:
            depth = depth[0, ...]

        if len_vids_shape == 5:
            vids = vids.squeeze(2)

        data[obs_key], data['depth'], data['tracks'] = vids, depth, tracks
        current_idx = len(file['data'])
        group = file.create_group(f'data/{current_idx}')
        create_dataset(group, **data)

    file.close()

def get_shape(data):
    if isinstance(data, np.ndarray):
        return data.shape
    
    if isinstance(data, List):
        return [get_shape(d) for d in data]
    
    if isinstance(data, Tuple):
        return tuple(get_shape(d) for d in data)
    
    if isinstance(data, Mapping):
        return {k: get_shape(v) for k, v in data.items()}
    
    return None

def create_dataset(h5_file: h5py.File, **kwargs):
    for key, value in kwargs.items():
        if isinstance(value, Mapping):
            group = h5_file.create_group(key)
            create_dataset(group, **value)
        else:
            h5_file.create_dataset(key, data=value)

def get_data(h5_file: h5py.File) -> Mapping[str, Any]:
    data = {}
    for key in h5_file.keys():
        if isinstance(h5_file[key], h5py.Group):
            data[key] = get_data(h5_file[key])
        else:
            data[key] = h5_file[key][:]

    return data

def get_values(h5_file: h5py.File, keys: Sequence[str]) -> Tuple[Any]:
    return [h5_file[key][:] for key in keys]

def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.numpy()
    
    if isinstance(value, List):
        return [to_numpy(v) for v in value]
    
    if isinstance(value, Mapping):
        return {k: to_numpy(v) for k, v in value.items()}
    
    return value

def to_tensor(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    
    if isinstance(value, List) or isinstance(value, Tuple):
        return [to_tensor(v) for v in value]
    
    if isinstance(value, Mapping):
        return {k: to_tensor(v) for k, v in value.items()}
    
    return torch.tensor(value)