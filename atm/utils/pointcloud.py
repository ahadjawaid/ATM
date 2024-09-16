from numpy import ndarray
import open3d as o3d
import numpy as np
from einops import rearrange
import torch

def get_depth_image(depth):
    return o3d.geometry.Image(depth)

def get_rgbd_image(rgb_img: ndarray, depth_img: ndarray) -> o3d.geometry.RGBDImage:
    rgb_img = o3d.geometry.Image(rgb_img)
    depth_img = o3d.geometry.Image(depth_img)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, depth_img)
    return rgbd_image

def get_point_cloud_from_rgbd(rgbd_image, intrinsic_matrix: ndarray, camera_height: int, camera_width: int) -> o3d.geometry.PointCloud:
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(camera_height, camera_width, intrinsic_matrix)
    )
    return point_cloud


def get_point_cloud_from_depth_torch(depth, intrinsic, rgb_values=None, depth_scalar=1):
    *_, channels, height, width = depth.shape
    dtype = depth.dtype

    if channels == 1 and width != 1:
        depth = rearrange(depth, '... c h w -> ... h w c')

    device = depth.device
    px, py = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])

    stacked_p = torch.tensor([[px, py],], dtype=dtype, device=device).unsqueeze(0)
    stacked_f = torch.tensor([[fx, fy],], dtype=dtype, device=device).unsqueeze(0)
    
    coordinates = torch.stack(torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device)), dim=-1).to(dtype)
    
    points = (((coordinates - stacked_p) * depth) / stacked_f)
    points = torch.concat([points, depth], dim=-1)
    points = rearrange(points, '... h w c -> ... (h w) c')

    if rgb_values is not None:
        *_, channels, height, width = rgb_values.shape
        if channels == 3 and width != 3:
            rgb_values = rearrange(rgb_values, '... c h w -> ... h w c')
        
        rgb_values = rearrange(rgb_values, '... h w c -> ... (h w) c')
        points = torch.concat([points, rgb_values], dim=-1)
        
    return points

def get_point_cloud_from_depth(depth, intrinsic_matrix: ndarray, camera_height: int, camera_width: int, depth_scalar: int = 1) -> o3d.geometry.PointCloud:
    if intrinsic_matrix.shape[0] > 3:
        intrinsic_matrix = intrinsic_matrix[0]
    
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
        depth,
        o3d.camera.PinholeCameraIntrinsic(camera_height, camera_width, intrinsic_matrix),
        depth_scale=depth_scalar
    )
    return point_cloud

def get_colored_point_cloud(point_cloud: o3d.geometry.PointCloud, rgb_values: ndarray) -> ndarray:
    point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if rgb_values.shape == 3:
        rgb_values = np.asarray(rgb_values).reshape(-1, 3)
    colored_points = np.concatenate((np.asarray(point_cloud.points), rgb_values), axis=1)
    return colored_points