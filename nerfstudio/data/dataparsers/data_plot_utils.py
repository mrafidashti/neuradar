import json
import numpy as np
import plotly.graph_objects as go

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lidars import Lidars, transform_points, transform_points_pairwise
from nerfstudio.cameras.radars import Radars
from nerfstudio.data.dataparsers.radar_eval_baseline import chamfer_distance
from nerfstudio.model_components.gospa import calculate_gospa
from nerfstudio.utils.poses import inverse, multiply, to4x4
from sklearn.cluster import DBSCAN

TEXT_MAP = ["x", "y", "z"]
RENDER_FILE_PATH = "/home/s0001331/renders/zod/test_render/train/radar_output/radar_data_1722935240.json"

def plot_pose(pose, fig=None, c="red", sensor_name=""):
    """
    Update a figure with the pose of an objet given the homogeneous transformation matrix.

    Inputs:
    - pose: 4x4 numpy array representing the homogeneous transformation matrix.

    Outputs:
    - fig: plotly figure object.
    """
    if fig is None:
        fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=[pose[0, 3]],
            y=[pose[1, 3]],
            z=[pose[2, 3]],
            mode="markers",
            marker=dict(size=5, color=c),
            name=f"{sensor_name} origin",
        )
    )
    for i in range(3):
        fig.add_trace(
            go.Scatter3d(
                x=[pose[0, 3], pose[0, 3] + pose[0, i]],
                y=[pose[1, 3], pose[1, 3] + pose[1, i]],
                z=[pose[2, 3], pose[2, 3] + pose[2, i]],
                mode="lines+text",
                line=dict(color=c),
                name=f"{TEXT_MAP[i]} axis",
                text=["", f"{TEXT_MAP[i]}", ""],
                textposition="bottom center"
            )
        )
    return fig

def plot_data_for_iteration(cameras: Cameras, lidars: Lidars, lidar_pointclouds, radars: Radars, radar_pointclouds, index: int = 0, render: bool = False, n_pcs: int = 0):
    # Get point clouds
    lidar_pc = transform_points_pairwise(lidar_pointclouds[index][:, :3].float(), lidars.lidar_to_worlds[index]).cpu().numpy()
    lidar_pc = lidar_pc[np.abs(lidar_pc[:, 0]) < 500]
    lidar_pc = lidar_pc[np.abs(lidar_pc[:, 1]) < 500]
    lidar_pc = lidar_pc[np.abs(lidar_pc[:, 2]) < 500]
    radar_index = np.argmin(np.abs(radars.times - lidars.times[index]))
    camera_index = np.argmin(np.abs(cameras.times - lidars.times[index]))
    radar_pc_r = radar_pointclouds[radar_index-n_pcs:radar_index+n_pcs+1]
    radar_pc_r = [pc[:, :3].float() for pc in radar_pc_r]
    r2w = radars.radar_to_worlds[radar_index-n_pcs:radar_index+n_pcs+1].cpu().numpy()
    ego_velocities = radars.metadata["velocities"][radar_index-n_pcs:radar_index+n_pcs+1].cpu().numpy()
    origin_radar = transform_points(torch.zeros(1, 3), r2w[n_pcs]).cpu().numpy()
    radar_pc = [transform_points(radar_pc_r[i], r2w[i]).cpu().numpy() for i in range(1+2*n_pcs)]

    import plotly.graph_objects as go
    # Plot the lidar_pc and radar_pc
    colors = ["red", "green", "purple", "orange", "pink", "cyan", "magenta", "brown", "black", "gray"]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=lidar_pc[:, 0], y=lidar_pc[:, 1], z=lidar_pc[:, 2], mode="markers", marker=dict(size=0.4, color='blue'), name="lidar"))

    for i in range(1+2*n_pcs):
        fig.add_trace(go.Scatter3d(x=radar_pc[i][:, 0], y=radar_pc[i][:, 1], z=radar_pc[i][:, 2], mode="markers", marker=dict(size=4, color=colors[i]), name=f"radar {i}"))

    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=2, color='magenta'), name="coords origin"))
    print(f"time diff lidar and radar: {lidars.times[index] - radars.times[radar_index]}")
    print(f"radar_index: {radar_index}, camera_index: {camera_index}")
    fig = plot_pose(to4x4(lidars.lidar_to_worlds[index]), fig=fig, c="blue", sensor_name="lidar")
    fig = plot_pose(to4x4(radars.radar_to_worlds[radar_index]), fig=fig, c=colors[n_pcs], sensor_name="radar")
    fig = plot_pose(to4x4(cameras.camera_to_worlds[camera_index]), fig=fig, c="green", sensor_name="camera")
    fig.update_layout(scene=dict(aspectmode="data"))
    fig.update_layout(xaxis_title="x (m)", yaxis_title="y (m)", scene_zaxis_title="z (m)")

    #radar_pointclouds[radar_index][:, :3].float()
    radial_unit_vectors = (radar_pc[n_pcs] - origin_radar) / np.linalg.norm(radar_pc[n_pcs] - origin_radar, axis=1, keepdims=True)
    range_rates = radar_pointclouds[radar_index][:, 4:5].float().cpu().numpy() / 5
    radial_velocity_vectors = radial_unit_vectors * range_rates
    for i in range(np.shape(radial_velocity_vectors)[0]):
        curr_pc = radar_pc[n_pcs][i, :3]
        vel_pc = curr_pc + radial_velocity_vectors[i, :3] + ego_velocities[n_pcs] / 5
        curr_pc = curr_pc.tolist()
        vel_pc = vel_pc.tolist()
        fig.add_trace(go.Scatter3d(x=[curr_pc[0], vel_pc[0]], y=[curr_pc[1], vel_pc[1]], z=[curr_pc[2], vel_pc[2]], mode="lines", line=dict(color=colors[n_pcs]), name=f"radar {n_pcs} velocity {i}"))

    if render:
        with open(RENDER_FILE_PATH, "r") as f:
            render_data = json.load(f)

        times = []
        for i in range(len(render_data)):
            times.append(render_data[i]["time"])

        render_index = np.argmin(np.abs(times - lidars.times[index].cpu().numpy()))
        render_data = render_data[render_index]
        sampled_points = render_data["mean_points"]
        sampled_points = np.array(sampled_points)
        transform = np.array(render_data["r2w"])
        rotations = transform[:3, :3]
        translations = transform[:3, 3].reshape((1, 3))
        sampled_points[:, :3] = sampled_points[:, :3] @ rotations.swapaxes(-2, -1) + translations
        indices = (np.array(render_data["existence_probabilities"]) > 0.5).flatten()
        fig.add_trace(go.Scatter3d(x=sampled_points[indices, 0], y=sampled_points[indices, 1], z=sampled_points[indices, 2], mode="markers", marker=dict(size=2, color='green'), name="rendered points"))

        radial_unit_vectors = (sampled_points[:, :3] - origin_radar) / np.linalg.norm(sampled_points[:, :3] - origin_radar, axis=1, keepdims=True)
        range_rates = sampled_points[:, -1].reshape((-1, 1)) / 10
        radial_velocity_vectors = radial_unit_vectors * range_rates
        for i in range(np.shape(radial_velocity_vectors)[0]):
            fig.add_trace(go.Scatter3d(x=[sampled_points[i, 0], sampled_points[i, 0] + radial_velocity_vectors[i, 0]], y=[sampled_points[i, 1], sampled_points[i, 1] + radial_velocity_vectors[i, 1]], z=[sampled_points[i, 2], sampled_points[i, 2] + radial_velocity_vectors[i, 2]], mode="lines", line=dict(color="green")))


    fig.show()

def get_metrics(radar_pc):

    for i in range(1, len(radar_pc)):
        print(f"Chamfer distances: {chamfer_distance(radar_pc[i-1], radar_pc[i])}")
        print(f"GOSPA: {calculate_gospa(radar_pc[i-1], radar_pc[i], c=1, p=1)[0]}")

def create_clusters(radar_pc, eps=1.0, min_samples=1):

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(radar_pc)

    # Create a dictionary to store the clusters
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(radar_pc[i])

    print(clusters)

    return clusters

def plot_clusters(clusters):
    fig = go.Figure()
    for label, cluster in clusters.items():
        cluster = np.array(cluster)
        fig.add_trace(go.Scatter3d(x=cluster[:, 0], y=cluster[:, 1], z=cluster[:, 2], mode="markers", marker=dict(size=4), name=f"Cluster {label}"))

    fig.update_layout(xaxis_title="x (m)", yaxis_title="y (m)", scene_zaxis_title="z (m)")
    fig.show()

def plot_radar_directions_from_spher_coords(directions_spher: torch.Tensor):
    # Convert to cartesian
    #directions_cart = torch.stack([directions_spher[:, 0] * torch.cos(directions_spher[:, 2]) * torch.cos(directions_spher[:, 1]),
    #                               directions_spher[:, 0] * torch.cos(directions_spher[:, 2]) * torch.sin(directions_spher[:, 1]),
    #                               directions_spher[:, 0] * torch.sin(directions_spher[:, 2])], dim=1)
    # Plot directions as lines
    directions_cart = directions_spher.cpu().detach().numpy()
    directions_cart = directions_cart[directions_cart[:, 0] < 250]
    directions_cart = directions_cart[directions_cart[:, 1] < 100]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=directions_cart[:, 0], y=directions_cart[:, 1], z=directions_cart[:, 2], mode="markers", marker=dict(size=1.5, color="blue")))
    #fig.update_layout(xaxis_title="x", yaxis_title="y", xaxis_range=[-10,250], yaxis_range=[-80,80])

    return fig
