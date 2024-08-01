import numpy as np
import torch

from atm.dataloader.base_dataset import BaseDataset, convert_points_to_tracks, convert_tracks_to_points
from atm.utils.flow_utils import sample_tracks_nearest_to_grids


class BCDataset(BaseDataset):
    def __init__(self, track_obs_fs=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_obs_fs = track_obs_fs

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = index - demo_start_index

        if self.cache_all:
            demo = self._cache[demo_id]
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                if self.cache_image:
                    all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
                else:
                    all_view_frames.append(self._load_image_list_from_disk(demo_id, view, time_offset))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_disk(demo_id, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
        else:
            demo_pth = self._demo_id_to_path[demo_id]
            demo = self.process_demo(self.load_h5(demo_pth))
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset))  # t c h w
                all_view_track_transformer_frames.append(
                    torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                )  # t tt_fs c h w

        all_view_tracks = []
        all_view_vis = []
        all_view_depths = []
        intrinsics = []
        for view in self.views:
            all_time_step_tracks = []
            all_time_step_vis = []
            all_time_step_depth = []
            for track_start_index in range(time_offset, time_offset+self.frame_stack):
                all_time_step_tracks.append(demo["root"][view]["tracks"][track_start_index:track_start_index + self.num_track_ts])  # track_len n 2
                all_time_step_vis.append(demo["root"][view]['vis'][track_start_index:track_start_index + self.num_track_ts])  # track_len n
                all_time_step_depth.append(demo["root"][view]["depth"][track_start_index:track_start_index + self.num_track_ts])
            all_view_tracks.append(torch.stack(all_time_step_tracks, dim=0))
            all_view_vis.append(torch.stack(all_time_step_vis, dim=0))
            all_view_depths.append(torch.stack(all_time_step_depth, dim=0))
            intrinsics.append(torch.from_numpy(demo['root'][view]['intrinsic']))

        obs = torch.stack(all_view_frames, dim=0)  # v t c h w
        track = torch.stack(all_view_tracks, dim=0)  # v t track_len n 2
        vi = torch.stack(all_view_vis, dim=0)  # v t track_len n
        depth = torch.stack(all_view_depths, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        track_transformer_obs = torch.stack(all_view_track_transformer_frames, dim=0)  # v t tt_fs c h w

        # augment rgbs and tracks
        if np.random.rand() < self.aug_prob:
            obs, track = self.augmentor((obs / 255., track))
            obs = obs * 255.

        # sample tracks
        sample_track, sample_vi = [], []
        for i in range(len(self.views)):
            sample_track_per_time, sample_vi_per_time = [], []
            for t in range(self.frame_stack):
                shape = obs.shape[-2:]
                curr_track = convert_points_to_tracks(track[i], intrinsics[i], shape) if self.use_points else track[i]
                track_i_t, vi_i_t = sample_tracks_nearest_to_grids(curr_track[t], vi[i, t], num_samples=self.num_track_ids)
                
                if self.use_points:
                    curr_depth = depth[i, t]
                    points = convert_tracks_to_points(track_i_t, curr_depth, intrinsics[i])
                    track_i_t = points

                sample_track_per_time.append(track_i_t)
                sample_vi_per_time.append(vi_i_t)
            sample_track.append(torch.stack(sample_track_per_time, dim=0))
            sample_vi.append(torch.stack(sample_vi_per_time, dim=0))
        track = torch.stack(sample_track, dim=0)
        vi = torch.stack(sample_vi, dim=0)

        actions = demo["root"]["actions"][time_offset:time_offset + self.frame_stack]
        task_embs = demo["root"]["task_emb_bert"]
        extra_states = {k: v[time_offset:time_offset + self.frame_stack] for k, v in
                        demo['root']['extra_states'].items()}
        return obs, track_transformer_obs, track, task_embs, actions, extra_states, depth, intrinsics
