import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from einops import rearrange
import os
from glob import glob
from natsort import natsorted
import h5py

from atm.dataloader.utils import load_rgb, ImgTrackColorJitter, ImgViewDiffTranslationAug


class BaseDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 img_size,
                 num_track_ts,
                 num_track_ids,
                 frame_stack=1,
                 cache_all=False,
                 cache_image=False,
                 num_demos=None,
                 vis=False,
                 aug_prob=0.,
                 augment_track=True,
                 views=None,
                 extra_state_keys=None,
                 use_points=False,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.vis = vis
        self.frame_stack = frame_stack
        self.num_demos = num_demos
        self.num_track_ts = num_track_ts
        self.num_track_ids = num_track_ids
        self.aug_prob = aug_prob
        self.augment_track = augment_track
        self.extra_state_keys = extra_state_keys
        self.cache_all = cache_all
        self.cache_image = cache_image
        self.use_points = use_points
        if not cache_all:
            assert not cache_image, "cache_image is only supported when cache_all is True."

        self.load_image_func = load_rgb

        self.views = views
        if self.views is not None:
            self.views.sort()

        if self.extra_state_keys is None:
            self.extra_state_keys = []

        if isinstance(self.dataset_dir, str):
            self.dataset_dir = [self.dataset_dir]

        self.buffer_fns = []
        for dir_idx, d in enumerate(self.dataset_dir):
            fn_list = glob(os.path.join(d, "*.hdf5"))
            fn_list = natsorted(fn_list)
            if self.num_demos is None:
                n_demo = len(fn_list)
            else:
                assert 0 < self.num_demos <= 1, "num_demos means the ratio of training data among all the demos."
                n_demo = int(len(fn_list) * self.num_demos)
            for fn in fn_list[:n_demo]:
                self.buffer_fns.append(fn)

        assert (len(self.buffer_fns) > 0)
        # print(f"found {len(self.buffer_fns)} trajectories in the specified folders: {self.dataset_dir}")

        self._cache = []
        self._index_to_demo_id, self._demo_id_to_path, self._demo_id_to_start_indices, self._demo_id_to_demo_length \
            = {}, {}, {}, {}
        self.load_demo_info()

        self.augmentor = transforms.Compose([
            ImgTrackColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            ImgViewDiffTranslationAug(input_shape=img_size, translation=8, augment_track=self.augment_track),
        ])

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

            demo_len = demo["root"][self.views[0]]["video"][0].shape[0]

            if self.cache_all:
                demo = self.process_demo(demo)
                if not self.cache_image:
                    for v in self.views:
                        del demo["root"][v]["video"]
                self._cache.append(demo)
            self._demo_id_to_path[demo_idx] = fn
            self._index_to_demo_id.update({k: demo_idx for k in range(start_idx, start_idx + demo_len)})
            self._demo_id_to_start_indices[demo_idx] = start_idx
            self._demo_id_to_demo_length[demo_idx] = demo_len
            start_idx += demo_len

        num_samples = len(self._index_to_demo_id)
        assert num_samples == start_idx

    def process_demo(self, demo):
        pad_length = self.frame_stack + self.num_track_ts
        for v in self.views:
            vids = demo["root"][v]['video'][0]  # t, c, h, w
            tracks = demo["root"][v]['tracks'][0]  # t, num_tracks, 2
            vis = demo["root"][v]['vis'][0]  # t, num_tracks
            depth = demo["root"][v]['depth'].transpose(0, 3, 1, 2) # t, c, h, w
            intrinsic = demo["root"][v]['intrinsic']
            t, c, h, w = vids.shape
            last_frame, last_track, last_vis, last_depth = vids[-1:], tracks[-1:], vis[-1:], depth[-1:]
            vids = np.concatenate([vids, np.repeat(last_frame, pad_length, axis=0)], axis=0)
            tracks = np.concatenate([tracks, np.repeat(last_track, pad_length, axis=0)], axis=0)
            vis = np.concatenate([vis, np.repeat(last_vis, pad_length, axis=0)], axis=0)
            depth = np.concatenate([depth, np.repeat(last_depth, pad_length, axis=0)], axis=0)
            vids, tracks, vis, depth = map(torch.Tensor, (vids, tracks, vis, depth))

            if self.use_points:
                points = convert_tracks_to_points(tracks, depth, intrinsic)

            # resize the images to the desired size.
            if h != self.img_size[0] or w != self.img_size[1]:
                vids = F.interpolate(vids, size=self.img_size, mode="bilinear", align_corners=False)

            demo["root"][v]['video'] = vids
            demo["root"][v]['tracks'] = tracks if not self.use_points else points
            demo["root"][v]['vis'] = vis
            demo["root"][v]['depth'] = depth
            demo['root'][v]['intrinsic'] = intrinsic

        actions = demo["root"]["actions"]
        if actions.ndim == 3:
            actions = actions[0]
        task_embs = demo["root"]["task_emb_bert"]
        extra_states_dict = {k: demo["root"]["extra_states"][k] for k in self.extra_state_keys}

        # pad action
        zero_action = np.zeros_like(actions[-1:])
        actions = np.concatenate([actions, np.repeat(zero_action, pad_length, axis=0)], axis=0)

        # pad extra states
        last_extra_states = {k: v[-1:] for k, v in extra_states_dict.items()}
        extra_states_dict = {
            k: np.concatenate([extra_states_dict[k], np.repeat(last_extra_states[k], pad_length, axis=0)], axis=0)
            for k in extra_states_dict
        }

        actions = torch.Tensor(actions)
        task_embs = torch.Tensor(task_embs)
        extra_states_dict = {k: torch.Tensor(v) for k, v in extra_states_dict.items()}

        demo["root"]['actions'] = actions
        demo['root']['task_emb_bert'] = task_embs
        demo['root']['extra_states'] = extra_states_dict

        return demo

    def _load_image_list_from_demo(self, demo, view, time_offset, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        if backward:
            image_indices = np.arange(max(time_offset + 1 - num_frames, 0), time_offset + 1)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length-1)
            frames = demo['root'][view]["video"][image_indices]
            if len(frames) < num_frames:
                padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
                frames = torch.cat([padding_frames, frames], dim=0)
            return frames
        else:
            return demo['root'][view]["video"][time_offset:time_offset + num_frames]

    def _load_image_list_from_disk(self, demo_id, view, time_offset, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames

        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        images_dir = os.path.join(demo_parent_dir, "images", demo_name)

        if backward:
            image_indices = np.arange(max(time_offset + 1 - num_frames, 0), time_offset + 1)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length-1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]
            frames = [np.zeros_like(frames[0]) for _ in range(num_frames - len(frames))] + frames  # padding with black images
        else:
            image_indices = np.arange(time_offset, time_offset + num_frames)
            image_indices = np.clip(image_indices, a_min=0, a_max=demo_length-1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]

        frames = np.stack(frames)  # t h w c
        frames = torch.Tensor(frames)
        frames = rearrange(frames, "t h w c -> t c h w")
        return frames

    def load_h5(self, fn):
        # return as a dict.
        def h5_to_dict(h5):
            d = {}
            for k, v in h5.items():
                if isinstance(v, h5py._hl.group.Group):
                    d[k] = h5_to_dict(v)
                else:
                    d[k] = np.array(v)
            return d

        with h5py.File(fn, 'r') as f:
            return h5_to_dict(f)

    def __len__(self):
        return len(self._index_to_demo_id)

    def __getitem__(self, index):
        raise NotImplementedError

def convert_tracks_to_points(tracks, depth, intrinsic):
    *_, n_time_steps, _, height, width = depth.shape
    device = tracks.device
    unnormalize_scalar = torch.tensor([[height, width],], dtype=torch.float32, device=device).unsqueeze(0)
    track_coordinates = (tracks * unnormalize_scalar).round().long()

    px, py = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])

    stacked_p = torch.tensor([[px, py],], dtype=torch.float32, device=device).unsqueeze(0)
    stacked_f = torch.tensor([[fx, fy],], dtype=torch.float32, device=device).unsqueeze(0)
    
    clamped_coordinates = torch.ones_like(track_coordinates, device=device)
    clamped_coordinates[..., 0] = track_coordinates[..., 0].clamp(0, height-1)
    clamped_coordinates[..., 1] = track_coordinates[..., 1].clamp(0, width-1)
    
    time_step_indicies = torch.arange(n_time_steps, device=device).unsqueeze(1)
    depth_values = depth[time_step_indicies, 0, clamped_coordinates[..., 0], clamped_coordinates[..., 1]].unsqueeze(-1)

    points = ((track_coordinates - stacked_p) / stacked_f) * depth_values
    points = torch.concatenate([points, depth_values], axis=-1)
    return points

def convert_points_to_tracks(points, intrinsic, shape):
    device = points.device
    px, py = intrinsic[0, 2], intrinsic[1, 2]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]

    stacked_p = torch.tensor([[px, py],], dtype=torch.float32, device=device).unsqueeze(0)
    stacked_f = torch.tensor([[fx, fy],], dtype=torch.float32, device=device).unsqueeze(0)

    depth_values = points[..., -1].unsqueeze(-1)
    point_coordinates = points[..., 0:-1]
    track_coordinates = (((point_coordinates / depth_values) * stacked_f) + stacked_p).long()

    height, width = shape
    normalize_scalar = torch.tensor([[height, width],], dtype=torch.float32, device=device).unsqueeze(0)
    tracks = track_coordinates / normalize_scalar

    return tracks