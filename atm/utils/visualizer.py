from atm.utils.pointcloud import get_depth_image, get_point_cloud_from_depth, get_colored_point_cloud
from flask import Flask, render_template_string
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from einops import rearrange
import plotly.io as pio
from PIL import Image
import open3d as o3d
import numpy as np
import torch
import io


class Visualizer:

    '''From the DP3 Repos Codebase'''
    def __init__(self):
        self.app = Flask(__name__)
        self.pointclouds = []
        
    def _generate_trace(self, pointcloud, color:tuple=None, size=5, opacity=0.7):
        x_coords = pointcloud[:, 0]
        y_coords = pointcloud[:, 1]
        z_coords = pointcloud[:, 2]

        if pointcloud.shape[1] == 3:
            if color is None:
                # design a colorful point cloud based on 3d coordinates
                # Normalize coordinates to range [0, 1]
                min_coords = pointcloud.min(axis=0)
                max_coords = pointcloud.max(axis=0)
                normalized_coords = (pointcloud - min_coords) / (max_coords - min_coords)
                try:
                    # Use normalized coordinates as RGB values
                    colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in normalized_coords]
                except: # maybe meet NaN error
                    # use simple cyan color
                    colors = ['rgb(0,255,255)' for _ in range(len(x_coords))]
            else:    
                colors = ['rgb({},{},{})'.format(color[0], color[1], color[2]) for _ in range(len(x_coords))]
        else:
            colors = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in pointcloud[:, 3:6]]

        return go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=size,
                opacity=opacity,
                color=colors
            )
        )


    def colorize(self, pointcloud):
        if pointcloud.shape[1] == 3:

            # design a colorful point cloud based on 3d coordinates
            # Normalize coordinates to range [0, 1]
            min_coords = pointcloud.min(axis=0)
            max_coords = pointcloud.max(axis=0)
            normalized_coords = (pointcloud - min_coords) / (max_coords - min_coords)
            try:
                # Use normalized coordinates as RGB values
                colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in normalized_coords]
            except: # maybe meet NaN error
                # use simple cyan color
                x_coords = pointcloud[:, 0]
                colors = ['rgb(0,255,255)' for _ in range(len(x_coords))]

        else:
            colors = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in pointcloud[:, 3:6]]
        return colors
    

    def visualize_pointcloud(self, pointcloud, color:tuple=None, camera_coordinate:tuple=(0, -0.001, 1.6)):
        trace = self._generate_trace(pointcloud, color=color, size=6, opacity=1.0)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=[trace], layout=layout)
        
        fig.update_layout(  
            scene=dict(
                # aspectmode='cube', 
                xaxis=dict(
                    showbackground=False,  # 隐藏背景网格
                    showgrid=True,        # 隐藏网格
                    showline=True,         # 显示轴线
                    linecolor='grey',      # 设置轴线颜色为灰色
                    zerolinecolor='grey',  # 设置0线颜色为灰色
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                    
                ),
                yaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                zaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                bgcolor='white',  # 设置背景色为白色
                camera=dict(eye=dict(zip(('x', 'y', 'z'), camera_coordinate)))
            )
        )
        div = pio.to_html(fig, full_html=False)

        @self.app.route('/')
        def index():
            return render_template_string('''<div>{{ div|safe }}</div>''', div=div)
        
        self.app.run(debug=True, use_reloader=False)

    def get_pointcloud(self, pointcloud, camera_coordinate:tuple=None, img_size=(512, 512), pixel_size=5) -> np.ndarray:
        trace = self._generate_trace(pointcloud, color=None, size=pixel_size, opacity=1.0)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=[trace], layout=layout)

        camera_coordinate = (0, -0.001, pointcloud[:, 2].max() * -2) if camera_coordinate is None else camera_coordinate

        fig.update_layout(  
            scene=dict(
                # aspectmode='cube', 
                xaxis=dict(
                    showbackground=False,  # Hide background grid
                    showgrid=True,         # Show grid
                    showline=True,         # Show axis line
                    linecolor='grey',      # Set axis line color to grey
                    zerolinecolor='grey',  # Set zero line color to grey
                    zeroline=False,        # Disable zero line
                    gridcolor='grey',      # Set grid color to grey
                    showticklabels=False,  # Hide axis numbers
                    title=''               # Remove axis label
                ),
                yaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,
                    gridcolor='grey',
                    showticklabels=False,  # Hide axis numbers
                    title=''               # Remove axis label
                ),
                zaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,
                    gridcolor='grey',
                    showticklabels=False,  # Hide axis numbers
                    title=''               # Remove axis label
                ),
                bgcolor='white',  # Set background color to white
                camera=dict(eye=dict(zip(('x', 'y', 'z'), camera_coordinate)))
            )
        )
                
        img_bytes = pio.to_image(fig, format="png", height=img_size[0], width=img_size[1])
        image = Image.open(io.BytesIO(img_bytes))
        image_np = np.array(image)

        if image_np.shape[-1] == 4:
            image_np = image_np[:, :, :3]

        return image_np

    def visualize_pointcloud_and_save(self, pointcloud, color:tuple=None, save_path=None):
        trace = self._generate_trace(pointcloud, color=color, size=6, opacity=1.0)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=[trace], layout=layout)
        
        fig.update_layout(
            
            scene=dict(
                # aspectmode='cube', 
                xaxis=dict(
                    showbackground=False,  # 隐藏背景网格
                    showgrid=True,        # 隐藏网格
                    showline=True,         # 显示轴线
                    linecolor='grey',      # 设置轴线颜色为灰色
                    zerolinecolor='grey',  # 设置0线颜色为灰色
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                    
                ),
                yaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                zaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                bgcolor='white'  # 设置背景色为白色
            )
        )
        # save
        fig.write_image(save_path, width=800, height=600)
        

    def save_visualization_to_file(self, pointcloud, file_path, color:tuple=None):
        # visualize pointcloud and save as html
        trace = self._generate_trace(pointcloud, color=color)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig_html = pio.to_html(go.Figure(data=[trace], layout=layout), full_html=True)

        with open(file_path, 'w') as file:
            file.write(fig_html)
        print(f"Visualization saved to {file_path}")


def get_colored_point_cloud_with_tracks(rgb_img: torch.Tensor, depth: torch.Tensor, tracks: torch.Tensor, intrinsic: torch.Tensor):
    depth = depth.to(torch.float32).numpy()
    rgb_img = rgb_img.squeeze().to(torch.uint8).numpy()
    tracks = tracks.squeeze().to(torch.float32).numpy()

    depth = rearrange(depth, '... c h w -> (... h) w c')
    depth_img = get_depth_image(np.ascontiguousarray(depth))
    point_cloud = get_point_cloud_from_depth(depth_img, intrinsic, *depth.shape[-2:])

    
    combined_point_cloud, expanded_colors = point_cloud, None
    for time_step in range(tracks.shape[0]):
        curr_track = tracks[time_step, :, :]

        track_point_cloud = o3d.geometry.PointCloud()
        track_point_cloud.points = o3d.utility.Vector3dVector(curr_track)

        combined_point_cloud = combined_point_cloud + track_point_cloud

        blue_color = np.array([time_step*10, 0, 150], dtype=np.uint8)
        expanded_color = np.tile(blue_color, (curr_track.shape[0], 1))

        if not isinstance(expanded_colors, np.ndarray):
            expanded_colors = expanded_color
        else:
            expanded_colors = np.concatenate([expanded_colors, expanded_color], axis=0)

    flattened_img = rearrange(rgb_img, '... c h w -> (... h w) c')
    combined_rgb_values = np.concatenate([flattened_img, expanded_colors], axis=0)

    colored_points = get_colored_point_cloud(combined_point_cloud, combined_rgb_values)
    return colored_points

def plot_2d_tracks(img: torch.Tensor, tracks: torch.Tensor) -> None:
    rgb_img = img.squeeze().long()
    if rgb_img.shape[0] == 3:
        rgb_img = rearrange(rgb_img, 'c h w -> h w c')
    for i in range(tracks.shape[0]):
        clamped_track_coordinates = (tracks[i, :, :] * 128).long().clamp(0, 128-1)
        blue_color = torch.tensor([i*10, 0, 150])
        for (x, y) in clamped_track_coordinates:
            rgb_img[x, y] = blue_color
    plt.imsave('tmp.png', rgb_img.numpy().astype(np.uint8))