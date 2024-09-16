import numpy as np

from atm.dataloader.base_dataset import BaseDataset, convert_tracks_to_points
from atm.utils.flow_utils import sample_tracks_visible_first
from torch.utils.data import Dataset
from atm.dataloader.utils import ImgViewDiffTranslationAug
import torchvision
from tqdm.auto import tqdm
import h5py
from typing import Sequence, Tuple, Any
from pathlib import Path
import torch


class ATMPretrainDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self._index_to_view_id = {}
        super().__init__(*args, **kwargs)

    def load_demo_info(self):
        start_idx = 0
        for demo_idx, fn in enumerate(self.buffer_fns):
            demo = self.load_h5(fn)

            if self.views is None:
                self.views = list(demo["root"].keys())
                self.views.remove("actions")
                self.views.remove("task_emb_bert")
                self.views.remove("extra_states")
                self.views.sort()
            try:
                demo_len = demo["root"][self.views[0]]["video"][0].shape[0]
            except Exception as e:
                print('demo_path:', fn, 'views:', self.views, 'demo["root"].keys()', demo["root"].keys(), 'demo.keys()', demo.keys())
                raise e

            if self.cache_all:
                demo = self.process_demo(demo)
                if not self.cache_image:
                    for v in self.views:
                        del demo["root"][v]["video"]  
                self._cache.append(demo)
            self._demo_id_to_path[demo_idx] = fn
            self._index_to_demo_id.update({k: demo_idx for k in range(start_idx, start_idx + demo_len*2)}) # *2 because two views
            self._index_to_view_id.update({k: (k - start_idx) % 2 for k in range(start_idx, start_idx + demo_len*2)})
            self._demo_id_to_start_indices[demo_idx] = start_idx
            self._demo_id_to_demo_length[demo_idx] = demo_len
            start_idx += demo_len * 2

        num_samples = len(self._index_to_demo_id)
        assert num_samples == start_idx

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        view = self.views[self._index_to_view_id[index]]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = (index - demo_start_index) // 2

        if self.cache_all:
            demo = self._cache[demo_id]
            if self.cache_image:
                vids = self._load_image_list_from_demo(demo, view, time_offset, backward=True)  # t c h w
            else:
                vids = self._load_image_list_from_disk(demo_id, view, time_offset, backward=True)  # t c h w
        else:
            demo_pth = self._demo_id_to_path[demo_id]
            demo = self.process_demo(self.load_h5(demo_pth))
            vids = self._load_image_list_from_demo(demo, view, time_offset, backward=True)  # t c h w

        tracks = demo["root"][view]["tracks"][time_offset:time_offset + self.num_track_ts]  # track_len n 2
        vis = demo["root"][view]['vis'][time_offset:time_offset + self.num_track_ts]  # track_len n
        task_emb = demo["root"]["task_emb_bert"]  # (dim,)
        intrinsic = demo['root'][view]['intrinsic']
        depth = demo['root'][view]['depth'][time_offset:time_offset + self.num_track_ts]   

        if self.use_points:
            tracks = convert_tracks_to_points(tracks, depth, intrinsic)

        # sample tracks
        tracks, vis = sample_tracks_visible_first(tracks, vis, num_samples=self.num_track_ids)

        depth = depth[0:1]

        return vids, tracks, vis, task_emb, intrinsic, depth


def create_cached_augmented_dataset(*args, **kwargs):
    atm_dataset = ATMPretrainDataset(*args, **kwargs)
    dataset_dir = Path(atm_dataset.dataset_dir[0]).parent.parent
    dataset_len = len(atm_dataset)
    img_size = atm_dataset.img_size
    aug_prob = atm_dataset.aug_prob
    augment_track = atm_dataset.augment_track

    file = h5py.File(dataset_dir/"atm_train_dataset.hdf5", "a")
    if 'data' not in file:
        file.create_group('data')
    
    if len(file['data']) < dataset_len:
        indices = range(len(file['data']), dataset_len)
        for idx in tqdm(indices, desc="Caching dataset"):
            vids, tracks, vis, task_emb, intrinsic, depth = atm_dataset.__getitem__(idx)

            current_idx = len(file['data'])
            group = file.create_group(f'data/{current_idx}')
            create_dataset(group, vids=vids, tracks=tracks, vis=vis, task_emb=task_emb, intrinsic=intrinsic, depth=depth)

    del atm_dataset

    color_augmentor = torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
    translation_augmentor = ImgViewDiffTranslationAug(input_shape=img_size, translation=8, augment_track=augment_track)
    
    
    n_augmentations = int((dataset_len * (1 / (1 - aug_prob))) - dataset_len) - (len(file['data']) - dataset_len)
    augmentation_indices = np.random.choice(dataset_len, n_augmentations)
    for idx in tqdm(augmentation_indices, desc="Augmenting dataset"):
        group = file[f'data/{idx}']
        vids, tracks, vis, task_emb, intrinsic, depth = get_values(group, ['vids', 'tracks', 'vis', 'task_emb', 'intrinsic', 'depth'])
        
        len_vids_shape, len_depth_shape, len_tracks_shape = len(vids.shape), len(depth.shape), len(tracks.shape)
        if len_vids_shape == 4:
            vids = vids[None]

        if len_depth_shape == 4:
            depth = depth[None]
        
        if len_tracks_shape == 3:
            tracks = tracks[None, None]


        vids, depth, tracks = map(lambda x: torch.from_numpy(x) if not isinstance(x, torch.Tensor) else x, 
                                  (vids, depth, tracks))
        
        vids = color_augmentor.forward(vids / 255.)
        vids, tracks, depth = translation_augmentor.forward((vids, tracks, depth))
        vids = vids * 255.
        
        if len_vids_shape == 4:
            vids = vids[0, ...]
        
        if len_depth_shape == 4:
            depth = depth[0, ...]

        if len_tracks_shape == 3:
            tracks = tracks[0, 0, ...]

        current_idx = len(file['data'])
        group = file.create_group(f'data/{current_idx}')
        create_dataset(group, vids=vids, tracks=tracks, vis=vis, task_emb=task_emb, intrinsic=intrinsic, depth=depth)

    file.close()

def create_dataset(h5_file: h5py.File, **kwargs):
    for key, value in kwargs.items():
        h5_file.create_dataset(key, data=value)

def get_values(h5_file: h5py.File, keys: Sequence[str]) -> Tuple[Any]:
    return tuple(h5_file[key][:] for key in keys)