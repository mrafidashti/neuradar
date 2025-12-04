# Copyright 2025 the authors of NeuRadar and contributors.
# Copyright 2025 the authors of NeuRAD and contributors.
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
render_radar.py
"""
from __future__ import annotations

import gzip
import json
import os
import pickle
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import List, Literal, Optional, Tuple, Union

import mediapy as media
import numpy as np
import plotly.graph_objects as go
import torch
import tyro
from jaxtyping import Float
from rich import box, style
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from torch import Tensor
from typing_extensions import Annotated

from nerfstudio.cameras.camera_paths import get_path_from_json
from nerfstudio.cameras.camera_utils import get_interpolated_poses
from nerfstudio.cameras.cameras import RayBundle
from nerfstudio.cameras.lidars import transform_points, transform_points_pairwise
from nerfstudio.cameras.radars import Radars, RadarType
from nerfstudio.data.datamanagers.ad_neuradar_datamanager import ADNeuRadarDataManager, ADNeuRadarDataManagerConfig
from nerfstudio.data.dataparsers.zod_dataparser import (
    RADAR_AZIMUTH_RAY_DIVERGENCE,
    RADAR_ELEVATION_RAY_DIVERGENCE,
    RADAR_FOV,
)
from nerfstudio.data.datasets.base_dataset import Dataset
from nerfstudio.data.datasets.radar_dataset import RadarDataset
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.model_components.radar_utils import (
    MultiBernoulli,
    add_actor_boxes_to_figure,
    plot_radar_samples,
    sample_radar_points,
)
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.scripts.render import BaseRender, modify_actors, plot_lidar_points
from nerfstudio.utils import colormaps
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.poses import inverse as pose_inverse, to4x4
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn


def update_json_file(filename, new_data):
    if os.path.exists(filename):
        # Load existing data
        with open(filename, "r") as file:
            existing_data = json.load(file)
    else:
        # If the file does not exist, start with an empty list
        existing_data = []

    # Append new data
    existing_data.append(new_data)

    # Write updated data back to the file
    with open(filename, "w") as file:
        json.dump(existing_data, file, indent=4)


def get_interpolated_radar_poses_many(
    poses: Float[Tensor, "num_poses 3 4"],
    steps_per_transition: int = 10,
    include_last: bool = True,
) -> Float[Tensor, "num_poses 3 4"]:
    """Return interpolated poses for many radar poses.

    Args:
        poses: list of radar poses
        steps_per_transition: number of steps per transition

    Returns:
        new poses
    """
    traj = []

    for idx in range(poses.shape[0] - 1):
        pose_a = poses[idx].cpu().numpy()
        pose_b = poses[idx + 1].cpu().numpy()
        poses_ab = get_interpolated_poses(pose_a, pose_b, steps=steps_per_transition, include_last=include_last)
        traj += poses_ab

    if not include_last:
        traj.append(poses[-1].cpu().numpy())

    traj = np.stack(traj, axis=0)

    return torch.tensor(traj, dtype=torch.float32)


def _render_radar_trajectory_video(
    pipeline: Pipeline,
    radar_path: Radars,
    output_filename: Path,
    output_format: Literal["images", "video"] = "images",
    actor_information: Optional[dict] = None,
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        crop_data: Crop data to apply to the rendered images.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
        depth_near_plane: Closest depth to consider when using the colormap for depth. If None, use min value.
        depth_far_plane: Furthest depth to consider when using the colormap for depth. If None, use max value.
        colormap_options: Options for colormap.
    """
    CONSOLE.print("[bold green]Creating trajectory " + output_format)
    radars = radar_path.to(pipeline.device)
    times = radar_path.times

    radars.metadata["velocities"] = torch.zeros_like(radars.radar_to_worlds[:, :3, 3])
    if len(radars) > 1:
        for sensor_idx in radars.metadata["sensor_idxs"].unique():
            mask = (radars.metadata["sensor_idxs"] == sensor_idx).squeeze(-1)
            radar2worlds, times = radars.radar_to_worlds[mask], radars.times[mask]
            delta_time = times[1:] - times[:-1]
            velo = (radar2worlds[1:, :3, 3] - radar2worlds[:-1, :3, 3]) / delta_time
            velo[velo.isnan()] = 0.0  # for overfitting
            radars.metadata["velocities"][mask] = torch.cat((velo, velo[-1:]), 0)

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(
            text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
            show_speed=True,
        ),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=False, compact=False),
        TimeElapsedColumn(),
    )
    output_image_dir = output_filename.parent / output_filename.stem
    output_image_dir.mkdir(parents=True, exist_ok=True)
    # NOTE:
    # we could use ffmpeg_args "-movflags faststart" for progressive download,
    # which would force moov atom into known position before mdat,
    # but then we would have to move all of mdat to insert metadata atom
    # (unless we reserve enough space to overwrite with our uuid tag,
    # but we don't know how big the video file will be, so it's not certain!)

    with progress:
        for radar_idx in progress.track(range(radars.size), description=""):
            radar_ray_bundle = radars.generate_rays(scan_indices=torch.tensor([radar_idx], device=pipeline.device))
            radar_ray_bundle.metadata.update({"is_radar": True})
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(radar_ray_bundle)
                radar_pcs = []
                pc, ber_indices = sample_radar_points(outputs["radar_output"], "nll")
                # pc = transform_points_pairwise(pc[:, :3], to4x4(radars.radar_to_worlds[radar_idx].to(pc.device)))
                radar_pcs.append(pc)
                mb = MultiBernoulli(outputs["radar_output"])
                figure = plot_radar_samples(None, pc, None, mb, ber_indices, None, "2d")

                # Add radar path to figure
                # radar_origin = radars.radar_to_worlds[radar_idx, :3, 3].cpu().numpy()
                radar_origin = np.zeros(3)
                figure.add_trace(
                    go.Scatter(
                        x=[radar_origin[0]], y=[radar_origin[1]], mode="markers", marker=dict(size=10, color="red")
                    )
                )

                if actor_information is not None:
                    figure = add_actor_boxes_to_figure(
                        figure, actor_information, radars.radar_to_worlds[radar_idx], radar_idx, "2d"
                    )

                # if radars.metadata["is_original_times"][radar_idx]:
                #    interp_status = " "
                # else:
                #    interp_status = "(Interpolated)"
                figure.update_layout(title=f"Radar Point Cloud at time {radars.times[radar_idx]}")
                figure.write_image(output_image_dir / f"{radar_idx:05d}.png", width=1600, height=1200)

    table = Table(
        title=None,
        show_header=False,
        box=box.MINIMAL,
        title_style=style.Style(bold=True),
    )
    table.add_row("Images", str(output_image_dir))
    CONSOLE.print(Panel(table, title="[bold][green]:tada: Render Complete :tada:[/bold]", expand=False))


@contextmanager
def _disable_datamanager_setup(cls):
    """
    Disables setup_train or setup_eval for faster initialization.
    """
    old_setup_train = getattr(cls, "setup_train")
    old_setup_eval = getattr(cls, "setup_eval")
    setattr(cls, "setup_train", lambda *args, **kwargs: None)
    setattr(cls, "setup_eval", lambda *args, **kwargs: None)
    yield cls
    setattr(cls, "setup_train", old_setup_train)
    setattr(cls, "setup_eval", old_setup_eval)


@dataclass
class RenderRadarFromCameraPath(BaseRender):
    """Render a camera path generated by the viewer or blender add-on."""

    camera_path_filename: Path = Path("camera_path.json")
    """Filename of the camera path to render."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=streamline_ad_config,
        )

        with open(self.camera_path_filename, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        # seconds = camera_path["seconds"]
        camera_path = get_path_from_json(camera_path)

        # Get Radars from camera path
        radars = Radars(
            radar_to_worlds=camera_path.camera_to_worlds,
            radar_type=RadarType.ContiARS40821,
            assume_ego_compensated=True,
            times=camera_path.times,
            metadata={"sensor_idxs": torch.zeros_like(camera_path.camera_type, dtype=torch.int64)},
            radar_azimuth_ray_divergence=RADAR_AZIMUTH_RAY_DIVERGENCE,
            radar_elevation_ray_divergence=RADAR_ELEVATION_RAY_DIVERGENCE,
            min_azimuth=RADAR_FOV[0][0],
            max_azimuth=RADAR_FOV[0][1],
            min_elevation=RADAR_FOV[1][0],
            max_elevation=RADAR_FOV[1][1],
        )

        _render_radar_trajectory_video(
            pipeline,
            radars,
            output_filename=self.output_path,
            output_format=self.output_format,
        )


@dataclass
class RadarInterpolatedRender(BaseRender):
    """Render a trajectory for interpolated poses between sensor poses."""

    output_path: Path = Path("renders")
    """Path to output video file."""
    downscale_factor: Optional[float] = None
    """Scaling factor to apply to the camera image resolution."""
    split: Literal["train", "eval", "train+eval"] = "train+eval"
    """Split to render."""
    frame_rate: int = 1
    """Frame rate of the output video."""
    sensor_idxs: Optional[List[int]] = None
    """Sensor indices to render. If None, render all sensors."""
    interpolation_steps: int = 4
    """Number of interpolation steps between eval dataset cameras."""

    def main(self) -> None:
        config: TrainerConfig

        def update_config(config: TrainerConfig) -> TrainerConfig:
            config = streamline_ad_config(config)
            data_manager_config = config.pipeline.datamanager
            assert isinstance(data_manager_config, ADNeuRadarDataManagerConfig)
            data_manager_config.eval_num_images_to_sample_from = -1
            data_manager_config.eval_num_times_to_repeat_images = -1
            data_manager_config.train_num_images_to_sample_from = -1
            data_manager_config.train_num_times_to_repeat_images = -1
            if self.downscale_factor is not None:
                assert hasattr(data_manager_config.dataparser, "downscale_factor")
                setattr(data_manager_config.dataparser, "downscale_factor", self.downscale_factor)
            # Remove any frame limit on the the dataparser
            config.pipeline.datamanager.dataparser.max_eval_frames = None
            return config

        config, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=update_config,
        )
        data_manager_config = config.pipeline.datamanager
        assert isinstance(data_manager_config, ADNeuRadarDataManagerConfig)

        for split in self.split.split("+"):
            datamanager: ADNeuRadarDataManager
            dataset: RadarDataset
            if split == "train":
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode="test", device=pipeline.device)

                dataset = datamanager.train_radar_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", datamanager.train_dataparser_outputs)
            else:
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode=split, device=pipeline.device)

                dataset = datamanager.eval_radar_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
                if dataparser_outputs is None:
                    dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(split=datamanager.test_split)

        radars = dataset.radars

        # sort radars based on time
        if radars.times is not None:
            radars = radars[torch.argsort(radars.times.squeeze(-1))]

        if radars.metadata and "sensor_idxs" in radars.metadata:
            sensor_idxs = (
                torch.tensor(self.sensor_idxs)
                if self.sensor_idxs is not None
                else radars.metadata["sensor_idxs"].unique()
            )
        else:
            sensor_idxs = torch.tensor(self.sensor_idxs) if self.sensor_idxs is not None else torch.tensor([0])
            if radars.metadata is None:
                radars.metadata = {}
            radars.metadata["sensor_idxs"] = torch.zeros_like(radars.radar_type, dtype=torch.int64)

        for sensor_i in sensor_idxs:
            sensor_i = sensor_i.item()
            curr_radars = radars[(radars.metadata["sensor_idxs"] == sensor_i).squeeze(-1)]

            new_poses = get_interpolated_radar_poses_many(
                poses=curr_radars.radar_to_worlds, steps_per_transition=self.interpolation_steps
            )

            radar_path_radar_to_worlds = new_poses

            if (times := curr_radars.times) is not None:
                radar_path_times = torch.from_numpy(
                    np.interp(
                        np.append(np.arange(0, len(times) - 1, 1 / self.interpolation_steps), len(times) - 1),
                        np.arange(len(times)),
                        times.squeeze(-1),
                    )[..., None]
                ).float()
                radar_path_times = radar_path_times[:-1]

            is_original_times = torch.tensor(
                [any(curr_radars.times == time) for time in radar_path_times], dtype=torch.bool
            ).view(-1, 1)

            radar_path_metadata = {"sensor_idxs": torch.full_like(radar_path_times, sensor_i)}
            radar_path_metadata.update({"is_original_times": is_original_times})
            radar_path_radar_azimuth_ray_divergence = torch.full_like(radar_path_times, RADAR_AZIMUTH_RAY_DIVERGENCE)
            radar_path_radar_elevation_ray_divergence = torch.full_like(
                radar_path_times, RADAR_ELEVATION_RAY_DIVERGENCE
            )
            radar_path_min_azimuth = torch.full_like(radar_path_times, RADAR_FOV[0][0])
            radar_path_max_azimuth = torch.full_like(radar_path_times, RADAR_FOV[0][1])
            radar_path_min_elevation = torch.full_like(radar_path_times, RADAR_FOV[1][0])
            radar_path_max_elevation = torch.full_like(radar_path_times, RADAR_FOV[1][1])
            radar_path_radar_type = torch.full_like(radar_path_times, curr_radars.radar_type[0, 0], dtype=torch.int)

            radar_path = Radars(
                radar_to_worlds=radar_path_radar_to_worlds,
                radar_type=radar_path_radar_type,
                assume_ego_compensated=True,
                times=radar_path_times,
                metadata=radar_path_metadata,
                radar_azimuth_ray_divergence=radar_path_radar_azimuth_ray_divergence,
                radar_elevation_ray_divergence=radar_path_radar_elevation_ray_divergence,
                min_azimuth=radar_path_min_azimuth,
                max_azimuth=radar_path_max_azimuth,
                min_elevation=radar_path_min_elevation,
                max_elevation=radar_path_max_elevation,
            )

            # Get info on dynamic actors
            actor_b2w, scan_indices, actor_indices = pipeline.model.dynamic_actors.get_boxes2world(
                query_times=radar_path_times.to(pipeline.device)
            )
            actor_sizes = pipeline.model.dynamic_actors.actor_sizes
            actor_information = {
                "actor_b2w": actor_b2w,
                "actor_indices": actor_indices,
                "actor_sizes": actor_sizes,
                "scan_indices": scan_indices,
            }

            # seconds = len(radar_path) / self.frame_rate

            output_filename = self.output_path.parent / f"{self.output_path.name}"
            _render_radar_trajectory_video(
                pipeline,
                radar_path,
                output_filename=output_filename,
                actor_information=actor_information,
            )


@dataclass
class RadarPoseShiftRender(BaseRender):
    """Render a trajectory where we shift the camera pose mid-way through."""

    shift: Tuple[float, ...] = (-2.0, 0.0, 0.0)
    """Shift to apply to the camera pose."""

    output_path: Path = Path("renders/")
    """Path to output video file."""
    downscale_factor: Optional[float] = None
    """Scaling factor to apply to the camera image resolution."""
    split: Literal["train", "eval", "train+eval"] = "train"
    """Split to render."""
    shift_time: float = 5
    """Time at which to apply the shift."""
    frame_rate: int = 1
    """Frame rate of the output video."""
    sensor_idxs: Optional[List[int]] = None
    """Sensor indices to render. If None, render all sensors."""
    shift_steps: int = 10
    """Number of steps to interpolate the shift over."""
    interpolation_steps: int = 4
    """Number of interpolation steps between eval dataset cameras."""

    def main(self):
        self.shift = torch.tensor(self.shift, dtype=torch.float32)

        config: TrainerConfig

        def update_config(config: TrainerConfig) -> TrainerConfig:
            config = streamline_ad_config(config)
            data_manager_config = config.pipeline.datamanager
            assert isinstance(data_manager_config, ADNeuRadarDataManagerConfig)
            data_manager_config.eval_num_images_to_sample_from = -1
            data_manager_config.eval_num_times_to_repeat_images = -1
            data_manager_config.train_num_images_to_sample_from = -1
            data_manager_config.train_num_times_to_repeat_images = -1
            if self.downscale_factor is not None:
                assert hasattr(data_manager_config.dataparser, "downscale_factor")
                setattr(data_manager_config.dataparser, "downscale_factor", self.downscale_factor)
            # Remove any frame limit on the the dataparser
            config.pipeline.datamanager.dataparser.max_eval_frames = None
            return config

        config, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=update_config,
        )
        data_manager_config = config.pipeline.datamanager
        assert isinstance(data_manager_config, ADNeuRadarDataManagerConfig)

        for split in self.split.split("+"):
            datamanager: ADNeuRadarDataManager
            dataset: RadarDataset
            if split == "train":
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode="test", device=pipeline.device)

                dataset = datamanager.train_radar_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", datamanager.train_dataparser_outputs)
            else:
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode=split, device=pipeline.device)

                dataset = datamanager.eval_radar_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
                if dataparser_outputs is None:
                    dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(split=datamanager.test_split)

        radars = dataset.radars

        # sort radars based on time
        if radars.times is not None:
            radars = radars[torch.argsort(radars.times.squeeze(-1))]

        if radars.metadata and "sensor_idxs" in radars.metadata:
            sensor_idxs = (
                torch.tensor(self.sensor_idxs)
                if self.sensor_idxs is not None
                else radars.metadata["sensor_idxs"].unique()
            )
        else:
            sensor_idxs = torch.tensor(self.sensor_idxs) if self.sensor_idxs is not None else torch.tensor([0])
            if radars.metadata is None:
                radars.metadata = {}
            radars.metadata["sensor_idxs"] = torch.zeros_like(radars.radar_type, dtype=torch.int64)

        for sensor_i in sensor_idxs:
            sensor_i = sensor_i.item()
            curr_radars = radars[(radars.metadata["sensor_idxs"] == sensor_i).squeeze(-1)]
            if curr_radars.times is not None:
                # find index of time closest to shift_time
                shift_idx = torch.argmin(torch.abs(curr_radars.times - self.shift_time))
            else:
                # warn user that we are assuming shift_time is the middle of the trajectory
                CONSOLE.print(
                    "Warning: Assuming shift_time is the middle of the trajectory. "
                    "If this is not the case, please specify times in the camera path JSON."
                )
                shift_idx = int(self.shift_time * len(curr_radars))
            pre_shift_radars = curr_radars[:shift_idx]
            post_shift_radars = curr_radars[shift_idx - 1 :]
            post_shift_radars.radar_to_worlds = post_shift_radars.radar_to_worlds.clone()
            post_shift_radars.radar_to_worlds[..., :3, 3] = post_shift_radars.radar_to_worlds[..., :3, 3] + self.shift
            mid_shift_radar_poses = get_interpolated_radar_poses_many(
                torch.cat([pre_shift_radars.radar_to_worlds[-1:], post_shift_radars.radar_to_worlds[:1]]),
                steps_per_transition=self.shift_steps,
            )

            mid_shift_radars = Radars(
                radar_to_worlds=mid_shift_radar_poses,
                radar_type=torch.full_like(
                    mid_shift_radar_poses[..., 0, 0], pre_shift_radars.radar_type[0, 0], dtype=torch.int
                ),
                assume_ego_compensated=True,
                metadata={
                    "sensor_idxs": torch.full_like(mid_shift_radar_poses[..., 0:1, 0], sensor_i),
                    "velocities": pre_shift_radars.metadata["velocities"][-1]
                    .view(1, 3)
                    .repeat(len(mid_shift_radar_poses[..., 0, 0]), 1),
                },
                radar_azimuth_ray_divergence=torch.full_like(
                    mid_shift_radar_poses[..., 0, 0], RADAR_AZIMUTH_RAY_DIVERGENCE
                ),
                radar_elevation_ray_divergence=torch.full_like(
                    mid_shift_radar_poses[..., 0, 0], RADAR_ELEVATION_RAY_DIVERGENCE
                ),
                min_azimuth=torch.full_like(mid_shift_radar_poses[..., 0, 0], RADAR_FOV[0][0]),
                max_azimuth=torch.full_like(mid_shift_radar_poses[..., 0, 0], RADAR_FOV[0][1]),
                min_elevation=torch.full_like(mid_shift_radar_poses[..., 0, 0], RADAR_FOV[1][0]),
                max_elevation=torch.full_like(mid_shift_radar_poses[..., 0, 0], RADAR_FOV[1][1]),
            )

            if (pre_shift_radars.times) is not None:
                mid_shift_radars.times = torch.full_like(
                    mid_shift_radar_poses[..., 0:1, 0], pre_shift_radars.times[-1].item()
                )

            radar_path = pre_shift_radars.cat([mid_shift_radars, post_shift_radars])
            radar_path.metadata = {"sensor_idxs": torch.full_like(radar_path.radar_type, sensor_i)}

            # Get info on dynamic actors
            actor_b2w, scan_indices, actor_indices = pipeline.model.dynamic_actors.get_boxes2world(
                query_times=radar_path.times.to(pipeline.device)
            )
            actor_sizes = pipeline.model.dynamic_actors.actor_sizes
            actor_information = {
                "actor_b2w": actor_b2w,
                "actor_indices": actor_indices,
                "actor_sizes": actor_sizes,
                "scan_indices": scan_indices,
            }

            # prepend sensor index to output filename
            output_filename = self.output_path.parent / self.output_path.name

            _render_radar_trajectory_video(
                pipeline,
                radar_path,
                output_filename=output_filename,
                actor_information=actor_information,
            )


@dataclass
class RadarActorRemovalRender(BaseRender):
    """Render a trajectory where we shift the camera pose mid-way through."""

    output_path: Path = Path("renders")
    """Path to output video file."""
    downscale_factor: Optional[float] = None
    """Scaling factor to apply to the camera image resolution."""
    split: Literal["train", "eval", "train+eval"] = "eval"
    """Split to render."""

    removal_time: float = 5
    """Time at which to apply the shift."""
    frame_rate: int = 1
    """Frame rate of the output video."""
    sensor_idxs: Optional[List[int]] = None
    """Sensor indices to render. If None, render all sensors."""
    interpolation_steps: int = 4
    """Number of interpolation steps between eval dataset cameras."""

    def main(self) -> None:
        config: TrainerConfig

        def update_config(config: TrainerConfig) -> TrainerConfig:
            config = streamline_ad_config(config)
            data_manager_config = config.pipeline.datamanager
            assert isinstance(data_manager_config, ADNeuRadarDataManagerConfig)
            data_manager_config.eval_num_images_to_sample_from = -1
            data_manager_config.eval_num_times_to_repeat_images = -1
            data_manager_config.train_num_images_to_sample_from = -1
            data_manager_config.train_num_times_to_repeat_images = -1
            if self.downscale_factor is not None:
                assert hasattr(data_manager_config.dataparser, "downscale_factor")
                setattr(data_manager_config.dataparser, "downscale_factor", self.downscale_factor)
            # Remove any frame limit on the the dataparser
            config.pipeline.datamanager.dataparser.max_eval_frames = None
            return config

        config, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=update_config,
        )
        data_manager_config = config.pipeline.datamanager
        assert isinstance(data_manager_config, ADNeuRadarDataManagerConfig)

        for split in self.split.split("+"):
            datamanager: ADNeuRadarDataManager
            dataset: RadarDataset
            if split == "train":
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode="test", device=pipeline.device)

                dataset = datamanager.train_radar_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", datamanager.train_dataparser_outputs)
            else:
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode=split, device=pipeline.device)

                dataset = datamanager.eval_radar_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
                if dataparser_outputs is None:
                    dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(split=datamanager.test_split)

        radars = dataset.radars

        # sort radars based on time
        if radars.times is not None:
            radars = radars[torch.argsort(radars.times.squeeze(-1))]

        if radars.metadata and "sensor_idxs" in radars.metadata:
            sensor_idxs = (
                torch.tensor(self.sensor_idxs)
                if self.sensor_idxs is not None
                else radars.metadata["sensor_idxs"].unique()
            )
        else:
            sensor_idxs = torch.tensor(self.sensor_idxs) if self.sensor_idxs is not None else torch.tensor([0])
            if radars.metadata is None:
                radars.metadata = {}
            radars.metadata["sensor_idxs"] = torch.zeros_like(radars.radar_type, dtype=torch.int64)

        for sensor_i in sensor_idxs:
            sensor_i = sensor_i.item()
            curr_radars = radars[(radars.metadata["sensor_idxs"] == sensor_i).squeeze(-1)]

            new_poses = get_interpolated_radar_poses_many(
                poses=curr_radars.radar_to_worlds, steps_per_transition=self.interpolation_steps
            )

            radar_path = curr_radars
            radar_path.radar_to_worlds = new_poses

            if (times := curr_radars.times) is not None:
                radar_path.times = torch.from_numpy(
                    np.interp(
                        np.append(np.arange(0, len(times) - 1, 1 / self.interpolation_steps), len(times) - 1),
                        np.arange(len(times)),
                        times.squeeze(-1),
                    )[..., None]
                ).float()
                radar_path.times = radar_path.times[:-1]

            radar_path.metadata = {"sensor_idxs": torch.full_like(radar_path.times, sensor_i)}
            radar_path.radar_azimuth_ray_divergence = torch.full_like(radar_path.times, RADAR_AZIMUTH_RAY_DIVERGENCE)
            radar_path.radar_elevation_ray_divergence = torch.full_like(
                radar_path.times, RADAR_ELEVATION_RAY_DIVERGENCE
            )
            radar_path.min_azimuth = torch.full_like(radar_path.times, RADAR_FOV[0][0])
            radar_path.max_azimuth = torch.full_like(radar_path.times, RADAR_FOV[0][1])
            radar_path.min_elevation = torch.full_like(radar_path.times, RADAR_FOV[1][0])
            radar_path.max_elevation = torch.full_like(radar_path.times, RADAR_FOV[1][1])
            radar_path.radar_type = torch.full_like(radar_path.times, radar_path.radar_type[0, 0], dtype=torch.int)

            no_actor_mask = pipeline.model.dynamic_actors.unique_timestamps > self.removal_time
            pipeline.model.dynamic_actors.actor_present_at_time[no_actor_mask, :] = False

            # Get info on dynamic actors
            actor_b2w, scan_indices, actor_indices = pipeline.model.dynamic_actors.get_boxes2world(
                query_times=radar_path.times.to(pipeline.device)
            )
            actor_sizes = pipeline.model.dynamic_actors.actor_sizes
            actor_information = {
                "actor_b2w": actor_b2w,
                "actor_indices": actor_indices,
                "actor_sizes": actor_sizes,
                "scan_indices": scan_indices,
            }

            output_filename = self.output_path.parent / f"{self.output_path.name}"
            _render_radar_trajectory_video(
                pipeline,
                radar_path,
                output_filename=output_filename,
                actor_information=actor_information,
            )


@dataclass
class RadarDatasetRender(BaseRender):
    """Render all images in the dataset."""

    output_path: Path = Path("renders")
    """Path to output video file."""
    data: Optional[Path] = None
    """Override path to the dataset."""
    config_output_dir: Optional[Path] = None
    """Override the config output dir. Used to load the model."""
    downscale_factor: Optional[float] = None
    """Scaling factor to apply to the camera image resolution."""
    split: Literal["train", "val", "test", "train+test"] = "train+test"
    """Split to render."""
    strict_load: bool = True
    """Whether to strictly load the config."""
    load_ignore_keys: Optional[List[str]] = field(
        default_factory=lambda: []
    )  # e.g. ["model.camera_optimizer.pose_adjustment", "_model.camera_optimizer.pose_adjustment"]
    """Keys to ignore when loading the config."""
    shift_loc_relative_to_cam: Tuple[float, float, float] = (
        0,
        0,
        0,
    )
    """Render images at this location (in nerfstudio cam coordinate system)."""

    def main(self):
        config: TrainerConfig

        def update_config(config: TrainerConfig) -> TrainerConfig:
            config = streamline_ad_config(config)
            data_manager_config = config.pipeline.datamanager
            assert isinstance(data_manager_config, ADNeuRadarDataManagerConfig)
            data_manager_config.eval_num_images_to_sample_from = -1
            data_manager_config.eval_num_times_to_repeat_images = -1
            data_manager_config.train_num_images_to_sample_from = -1
            data_manager_config.train_num_times_to_repeat_images = -1
            if self.data is not None:
                data_manager_config.data = self.data
            if self.config_output_dir is not None:
                config.output_dir = self.config_output_dir
            if self.downscale_factor is not None:
                assert hasattr(data_manager_config.dataparser, "downscale_factor")
                setattr(data_manager_config.dataparser, "downscale_factor", self.downscale_factor)
            # Remove any frame limit on the the dataparser
            config.pipeline.datamanager.dataparser.max_eval_frames = None
            return config

        config, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=update_config,
            strict_load=self.strict_load,
            ignore_keys=self.load_ignore_keys,
        )
        data_manager_config = config.pipeline.datamanager
        assert isinstance(data_manager_config, ADNeuRadarDataManagerConfig)

        for split in self.split.split("+"):
            datamanager: ADNeuRadarDataManager
            dataset: RadarDataset
            if split == "train":
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode="test", device=pipeline.device)

                dataset = datamanager.train_radar_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", datamanager.train_dataparser_outputs)
            else:
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode=split, device=pipeline.device)

                dataset = datamanager.eval_radar_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
                if dataparser_outputs is None:
                    dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(split=datamanager.test_split)

            shift_relative_to_cam = torch.tensor(self.shift_loc_relative_to_cam, dtype=torch.float32)
            # add homogenous point
            shift_relative_to_cam = torch.cat([shift_relative_to_cam, torch.tensor([1.0], dtype=torch.float32)])
            shift_relative_to_cam = shift_relative_to_cam.to(dataset.radars.radar_to_worlds.device)
            # shift the camera poses
            dataset.radars.radar_to_worlds[..., :3, 3:4] = (
                dataset.radars.radar_to_worlds @ shift_relative_to_cam.reshape(1, 4, 1)
            )

            dataloader = FixedIndicesEvalDataloader(
                dataset=dataset,
                device=datamanager.device,
                num_workers=datamanager.world_size * 4,
            )

            json_file = self.output_path / split / self.rendered_output_names[0] / f"radar_data_{int(time())}.json"

            with Progress(
                TextColumn(f":movie_camera: Rendering split {split} :movie_camera:"),
                BarColumn(),
                TaskProgressColumn(
                    text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                    show_speed=True,
                ),
                ItersPerSecColumn(suffix="fps"),
                TimeRemainingColumn(elapsed_when_finished=False, compact=False),
                TimeElapsedColumn(),
            ) as progress:
                for camera_idx, (ray_bundle, batch) in enumerate(progress.track(dataloader, total=len(dataset))):
                    ray_bundle: RayBundle
                    with torch.no_grad():
                        ray_bundle.metadata.update({"is_radar": True})
                        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)

                    r2w = dataset.radars.radar_to_worlds[camera_idx].to(pipeline.device)
                    sensor_idx = batch["sensor_idx"]

                    gt_batch = batch.copy()["radar"][:, :3].to(pipeline.device)

                    rendered_output, ber_indices = sample_radar_points(outputs["radar_output"], "nll")
                    rendered_output = transform_points_pairwise(rendered_output[:, :3], to4x4(r2w))
                    # mean_points = sample_radar_points(outputs["radar_output"], "euclidean",  existence_threshold=EP_THRESHOLD).squeeze(0)
                    # transformed_points = transform_points_pairwise(rendered_output, r2w)
                    mb = MultiBernoulli(outputs["radar_output"])
                    figure = plot_radar_samples(gt_batch, rendered_output, None, mb, ber_indices, None, "2d")

                    # Get info on dynamic actors
                    actor_b2w, scan_indices, actor_indices = pipeline.model.dynamic_actors.get_boxes2world(
                        query_times=dataset.radars.times.to(pipeline.device)
                    )
                    actor_sizes = pipeline.model.dynamic_actors.actor_sizes
                    actor_information = {
                        "actor_b2w": actor_b2w.to(pipeline.device),
                        "actor_indices": actor_indices.to(pipeline.device),
                        "actor_sizes": actor_sizes.to(pipeline.device),
                        "scan_indices": scan_indices.to(pipeline.device),
                    }

                    figure = add_actor_boxes_to_figure(
                        figure,
                        actor_information,
                        dataset.radars.to(pipeline.device).radar_to_worlds[camera_idx],
                        camera_idx,
                        "2d",
                    )

                    rendered_output_name = self.rendered_output_names[0]

                    if rendered_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                        CONSOLE.print(
                            f"Please set --rendered-output-name to one of: {outputs.keys()}", justify="center"
                        )
                        sys.exit(1)

                    image_name = f"{camera_idx:05d}"

                    # Try to get the original filename

                    output_path = self.output_path / split / rendered_output_name / f"sensor_{sensor_idx}" / image_name
                    output_path.parent.mkdir(exist_ok=True, parents=True)

                    del rendered_output_name

                    # Save to file
                    figure.write_image(output_path.parent / (output_path.name + ".jpg"), width=1600, height=1200)

                    update_json_file(
                        json_file,
                        {
                            "filename": str(output_path.parent / (output_path.name + ".jpg")),
                            "sensor_idx": sensor_idx.cpu().numpy().tolist(),
                            "camera_idx": [camera_idx],
                            "gt_points": gt_batch.cpu().numpy().tolist(),
                            "rendered_points": rendered_output.cpu().numpy().tolist(),
                            "existence_probabilities": mb.existence_probabilities.cpu().numpy().tolist(),
                            "r2w": r2w.cpu().numpy().tolist(),
                            "time": dataset.radars.times[camera_idx].cpu().numpy().tolist(),
                        },
                    )

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        for split in self.split.split("+"):
            table.add_row(f"Outputs {split}", str(self.output_path / split))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Render on split {} Complete :tada:[/bold]", expand=False))


@dataclass
class FullSensorSetRender(BaseRender):
    """Render all images in the dataset."""

    pose_source: Literal["train", "val", "test", "train+test", "train+val"] = "test"
    """Split to render."""
    output_path: Path = Path("renders")
    """Path to output video file."""
    data: Optional[Path] = None
    """Override path to the dataset."""
    config_output_dir: Optional[Path] = None
    """Override the config output dir. Used to load the model."""
    downscale_factor: Optional[float] = None
    """Scaling factor to apply to the camera image resolution."""
    rendered_output_names: List[str] = field(default_factory=lambda: ["all"])
    """Name of the renderer outputs to use. rgb, depth, raw-depth, gt-rgb etc. By default all outputs are rendered."""
    strict_load: bool = True
    """Whether to strictly load the config."""
    load_ignore_keys: Optional[List[str]] = field(
        default_factory=lambda: []
    )  # e.g. ["model.camera_optimizer.pose_adjustment", "_model.camera_optimizer.pose_adjustment"]
    """Keys to ignore when loading the config."""

    render_height: Optional[int] = None
    """Height to render the images at."""
    render_width: Optional[int] = None
    """Width to render the images at."""
    output_height: Optional[int] = None
    """Height to crop the output images at."""
    output_width: Optional[int] = None
    """Width to crop the output images at."""

    shift: Tuple[float, float, float] = (0, 0, 0)
    """Shift to apply to the camera pose."""

    actor_shift: Tuple[float, ...] = (0.0, 0.0, 0.0)
    """Shift to apply to all actor poses."""
    actor_removal_time: Optional[float] = None
    """Time at which to remove all actors."""
    actor_stop_time: Optional[float] = None
    """Time at which to stop all actors."""
    actor_indices: Optional[List[int]] = None
    """Indices of actors to modify. If None, modify all actors."""

    calculate_and_save_metrics: bool = True
    """Whether to calculate and save metrics."""
    metrics_filename: Path = Path("metrics.pkl")
    """Filename to save the metrics to."""

    render_point_clouds: bool = True
    """Whether to render point clouds."""

    render_radar: bool = True
    """Whether to render radar point cloud."""

    def main(self):
        config: TrainerConfig

        def update_config(config: TrainerConfig) -> TrainerConfig:
            config = streamline_ad_config(config)
            data_manager_config = config.pipeline.datamanager
            assert isinstance(data_manager_config, (ADNeuRadarDataManagerConfig))
            data_manager_config.eval_num_images_to_sample_from = -1
            data_manager_config.eval_num_times_to_repeat_images = -1
            if isinstance(data_manager_config, ADNeuRadarDataManagerConfig):
                data_manager_config.train_num_images_to_sample_from = -1
                data_manager_config.train_num_times_to_repeat_images = -1
            if self.data is not None:
                data_manager_config.data = self.data
            if self.config_output_dir is not None:
                config.output_dir = self.config_output_dir
            if self.downscale_factor is not None:
                assert hasattr(data_manager_config.dataparser, "downscale_factor")
                setattr(data_manager_config.dataparser, "downscale_factor", self.downscale_factor)
            # Remove any frame limit on the the dataparser
            config.pipeline.datamanager.dataparser.max_eval_frames = None
            return config

        config, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=update_config,
            strict_load=self.strict_load,
            ignore_keys=self.load_ignore_keys,
        )
        data_manager_config = config.pipeline.datamanager
        assert isinstance(data_manager_config, (ADNeuRadarDataManagerConfig))

        modify_actors(pipeline, self.actor_shift, self.actor_removal_time, self.actor_stop_time, self.actor_indices)

        self.output_path.mkdir(exist_ok=True, parents=True)
        metrics_out = dict()
        for split in self.pose_source.split("+"):
            datamanager: ADNeuRadarDataManager
            dataset: Dataset
            if split == "train":
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode="test", device=pipeline.device)

                dataset = datamanager.train_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", datamanager.train_dataparser_outputs)
                lidar_dataset = datamanager.train_lidar_dataset
                radar_dataset = datamanager.train_radar_dataset
            else:
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode=split, device=pipeline.device)

                dataset = datamanager.eval_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
                lidar_dataset = datamanager.eval_lidar_dataset
                radar_dataset = datamanager.eval_radar_dataset
                if dataparser_outputs is None:
                    dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(split=datamanager.test_split)
            dataset.cameras.height = (
                torch.full_like(dataset.cameras.height, self.render_height)
                if self.render_height is not None
                else dataset.cameras.height
            )
            dataset.cameras.width = (
                torch.full_like(dataset.cameras.width, self.render_width)
                if self.render_width is not None
                else dataset.cameras.width
            )
            shift_relative_to_cam = torch.tensor(self.shift, dtype=torch.float32)
            # add homogenous point
            shift_relative_to_cam = torch.cat([shift_relative_to_cam, torch.tensor([1.0], dtype=torch.float32)])
            shift_relative_to_cam = shift_relative_to_cam.to(dataset.cameras.camera_to_worlds.device)
            # shift the camera poses
            dataset.cameras.camera_to_worlds[..., :3, 3:4] = (
                dataset.cameras.camera_to_worlds @ shift_relative_to_cam.reshape(1, 4, 1)
            )

            dataloader = FixedIndicesEvalDataloader(
                dataset=dataset,
                device=datamanager.device,
                num_workers=datamanager.world_size * 4,
            )
            lidar_dataloader = FixedIndicesEvalDataloader(
                dataset=lidar_dataset,
                device=datamanager.device,
                num_workers=datamanager.world_size * 4,
            )
            radar_dataloader = FixedIndicesEvalDataloader(
                dataset=radar_dataset,
                device=datamanager.device,
                num_workers=datamanager.world_size * 4,
            )
            images_root = Path(os.path.commonpath(dataparser_outputs.image_filenames))
            with Progress(
                TextColumn(f":movie_camera: Rendering split {split} :movie_camera:"),
                BarColumn(),
                TaskProgressColumn(
                    text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                    show_speed=True,
                ),
                ItersPerSecColumn(suffix="fps"),
                TimeRemainingColumn(elapsed_when_finished=False, compact=False),
                TimeElapsedColumn(),
            ) as progress:
                for camera_idx, (camera, batch) in enumerate(progress.track(dataloader, total=len(dataset))):
                    # Try to get the original filename
                    image_name = (
                        Path(dataparser_outputs.image_filenames[camera_idx]).with_suffix("").relative_to(images_root)
                    )

                    with torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(
                            camera.generate_rays(camera_indices=0, keep_shape=True)
                        )

                    if self.output_height is not None:
                        dataset.cameras.height[batch["image_idx"]] = torch.full_like(
                            dataset.cameras.height[0:1], self.output_height
                        )
                        batch["image"] = batch["image"][..., : self.output_height, :, :]
                        outputs["rgb"] = outputs["rgb"][..., : self.output_height, :, :]

                    if self.output_width is not None:
                        dataset.cameras.width[batch["image_idx"]] = torch.full_like(
                            dataset.cameras.width[0:1], self.output_width
                        )
                        batch["image"] = batch["image"][..., : self.output_width, :]
                        outputs["rgb"] = outputs["rgb"][..., : self.output_width, :]

                    if self.calculate_and_save_metrics:
                        with torch.no_grad():
                            metrics_dict, _, _ = pipeline.model.get_image_metrics_and_images(outputs, batch)
                            metrics_out[str(image_name)] = metrics_dict

                    gt_batch = batch.copy()
                    gt_batch["rgb"] = gt_batch.pop("image")
                    all_outputs = (
                        list(outputs.keys())
                        + [f"raw-{x}" for x in outputs.keys()]
                        + [f"gt-{x}" for x in gt_batch.keys()]
                        + [f"raw-gt-{x}" for x in gt_batch.keys()]
                    )
                    rendered_output_names = self.rendered_output_names
                    if "all" in rendered_output_names:
                        rendered_output_names = ["gt-rgb"] + list(outputs.keys())
                    elif rendered_output_names == ["none"]:
                        rendered_output_names = []
                    if "dir_fig" in rendered_output_names:
                        rendered_output_names.remove("dir_fig")

                    for rendered_output_name in rendered_output_names:
                        if rendered_output_name not in all_outputs:
                            CONSOLE.rule("Error", style="red")
                            CONSOLE.print(
                                f"Could not find {rendered_output_name} in the model outputs", justify="center"
                            )
                            CONSOLE.print(
                                f"Please set --rendered-output-name to one of: {all_outputs}", justify="center"
                            )
                            sys.exit(1)

                        is_raw = False
                        is_depth = rendered_output_name.find("depth") != -1

                        output_path = self.output_path / split / rendered_output_name / image_name
                        output_path.parent.mkdir(exist_ok=True, parents=True)

                        output_name = rendered_output_name
                        if output_name.startswith("raw-"):
                            output_name = output_name[4:]
                            is_raw = True
                            if output_name.startswith("gt-"):
                                output_name = output_name[3:]
                                output_image = gt_batch[output_name]
                            else:
                                output_image = outputs[output_name]
                                if is_depth:
                                    # Divide by the dataparser scale factor
                                    output_image.div_(dataparser_outputs.dataparser_scale)
                        else:
                            if output_name.startswith("gt-"):
                                output_name = output_name[3:]
                                output_image = gt_batch[output_name]
                            else:
                                output_image = outputs[output_name]
                        del output_name

                        # Map to color spaces / numpy
                        if is_raw:
                            output_image = output_image.cpu().numpy()
                        elif is_depth:
                            output_image = (
                                colormaps.apply_depth_colormap(
                                    output_image,
                                    accumulation=outputs["accumulation"],
                                    near_plane=self.depth_near_plane,
                                    far_plane=self.depth_far_plane,
                                    colormap_options=self.colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )
                        else:
                            output_image = (
                                colormaps.apply_colormap(
                                    image=output_image,
                                    colormap_options=self.colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )

                        # Save to file
                        height = (
                            min(output_image.shape[0], self.output_height)
                            if self.output_height
                            else output_image.shape[0]
                        )
                        width = (
                            min(output_image.shape[1], self.output_width)
                            if self.output_width
                            else output_image.shape[1]
                        )
                        output_image = output_image[:height, :width]
                        if is_raw:
                            with gzip.open(output_path.parent / (output_path.name + ".npy.gz"), "wb") as f:
                                np.save(f, output_image)
                        elif self.image_format == "png":
                            media.write_image(output_path.parent / (output_path.name + ".png"), output_image, fmt="png")
                        elif self.image_format == "jpeg":
                            media.write_image(
                                output_path.parent / (output_path.name + ".jpg"),
                                output_image,
                                fmt="jpeg",
                                quality=self.jpeg_quality,
                            )
                        else:
                            raise ValueError(f"Unknown image format {self.image_format}")

            if self.render_point_clouds:
                with Progress(
                    TextColumn(f":movie_camera: Rendering lidars for split {split} :movie_camera:"),
                    BarColumn(),
                    TaskProgressColumn(
                        text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                        show_speed=True,
                    ),
                    ItersPerSecColumn(suffix="fps"),
                    TimeRemainingColumn(elapsed_when_finished=False, compact=False),
                    TimeElapsedColumn(),
                ) as progress:
                    with torch.no_grad():
                        output_path = self.output_path / split / "lidar"
                        output_path.mkdir(exist_ok=True, parents=True)
                        for lidar_idx, (lidar, batch) in enumerate(
                            progress.track(lidar_dataloader, total=len(lidar_dataloader))
                        ):
                            points = batch["lidar"]
                            lidar_indices = torch.zeros_like(points[:, 0:1]).long()
                            ray_bundle = lidar.generate_rays(
                                lidar_indices=lidar_indices, points=points, keep_shape=True
                            )
                            batch["is_lidar"] = ray_bundle.metadata["is_lidar"]
                            batch["distance"] = ray_bundle.metadata["directions_norm"]
                            batch["did_return"] = ray_bundle.metadata["did_return"]

                            lidar_output = pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)

                            # add points in local coords to model outputs
                            l2w = lidar.lidar_to_worlds[0].to(pipeline.device)
                            w2l = pose_inverse(l2w)
                            points = ray_bundle.origins + ray_bundle.directions * lidar_output["depth"]
                            lidar_output["points"] = (
                                w2l @ torch.cat([points, torch.ones_like(points[..., :1])], dim=-1).unsqueeze(-1)
                            ).squeeze(-1)

                            points_in_local = lidar_output["points"]
                            if "ray_drop_prob" in lidar_output:
                                points_in_local = points_in_local[(lidar_output["ray_drop_prob"] < 0.5).squeeze(-1)]

                            points_in_world = transform_points(points_in_local, lidar.lidar_to_worlds[0])
                            # get ground truth for comparison
                            gt_point_in_world = transform_points(batch["lidar"][..., :3], lidar.lidar_to_worlds[0])
                            plot_lidar_points(
                                gt_point_in_world.cpu().detach().numpy(), output_path / f"gt-lidar_{lidar_idx}.png"
                            )
                            plot_lidar_points(
                                points_in_world.cpu().detach().numpy(), output_path / f"lidar_{lidar_idx}.png"
                            )

            if self.render_radar:
                with Progress(
                    TextColumn(f":movie_camera: Rendering radar for split {split} :movie_camera:"),
                    BarColumn(),
                    TaskProgressColumn(
                        text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                        show_speed=True,
                    ),
                    ItersPerSecColumn(suffix="fps"),
                    TimeRemainingColumn(elapsed_when_finished=False, compact=False),
                    TimeElapsedColumn(),
                ) as progress:
                    with torch.no_grad():
                        output_path = self.output_path / split / "radar"
                        output_path.mkdir(exist_ok=True, parents=True)
                        for radar_idx, (radar, batch) in enumerate(
                            progress.track(radar_dataloader, total=len(radar_dataloader))
                        ):
                            pipeline.model.set_active_levels(radar_idx)
                            radar_idx = batch["radar_idx"]
                            num_points = batch["radar"].shape[0]
                            ray_indices = torch.cat(
                                [
                                    torch.full((num_points, 1), radar_idx, dtype=torch.int64, device=pipeline.device),
                                    torch.arange(num_points, device=pipeline.device).view(-1, 1),
                                ],
                                dim=-1,
                            )
                            batch["indices"] = ray_indices
                            ray_bundle = radar.generate_rays(scan_indices=torch.tensor([0], device=pipeline.device)).to(
                                pipeline.device
                            )

                            is_radar = torch.ones((len(ray_bundle), 1), dtype=torch.bool, device=pipeline.device)
                            ray_bundle.metadata["is_radar"], batch["is_radar"] = is_radar, is_radar
                            batch["distance"] = ray_bundle.metadata["directions_norm"]
                            batch["did_return"] = ray_bundle.metadata["did_return"]

                            radar_output = pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)

                            if self.calculate_and_save_metrics:
                                with torch.no_grad():
                                    metrics_dict, _, _ = pipeline.model.get_image_metrics_and_images(
                                        radar_output, batch
                                    )
                                    metrics_out["radar_" + str(radar_idx)] = metrics_dict

                            rendered_output, ber_indices = sample_radar_points(
                                radar_output["radar_output"], pipeline.model.config.loss.radar_loss_type
                            )
                            # r2w = radar_dataset.radars.radar_to_worlds[radar_idx].to(pipeline.device)
                            # rendered_output = transform_points_pairwise(rendered_output[:, :3], to4x4(r2w))
                            mb = MultiBernoulli(radar_output["radar_output"])
                            gt_batch = batch.copy()["radar"][:, :3].to(pipeline.device)
                            figure = plot_radar_samples(gt_batch, rendered_output, None, mb, ber_indices, None, "2d")

                            # Get info on dynamic actors
                            actor_b2w, scan_indices, actor_indices = pipeline.model.dynamic_actors.get_boxes2world(
                                query_times=radar_dataset.radars.times.to(pipeline.device)
                            )
                            actor_sizes = pipeline.model.dynamic_actors.actor_sizes
                            actor_information = {
                                "actor_b2w": actor_b2w.to(pipeline.device),
                                "actor_indices": actor_indices.to(pipeline.device),
                                "actor_sizes": actor_sizes.to(pipeline.device),
                                "scan_indices": scan_indices.to(pipeline.device),
                            }

                            figure = add_actor_boxes_to_figure(
                                figure,
                                actor_information,
                                radar_dataset.radars.to(pipeline.device).radar_to_worlds[radar_idx],
                                radar_idx,
                                "2d",
                            )

                            # Save to file
                            figure.write_image(output_path / (str(radar_idx) + ".jpg"), width=1800, height=1400)

        if self.calculate_and_save_metrics:
            metrics_out_path = Path(self.output_path, self.metrics_filename)
            with open(metrics_out_path, "wb") as f:
                pickle.dump(metrics_out, f)
            CONSOLE.print(f"[bold][green]:glowing_star: Metrics saved to {metrics_out_path}")

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        for split in self.pose_source.split("+"):
            table.add_row(f"Outputs {split}", str(self.output_path / split))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Render on split {} Complete :tada:[/bold]", expand=False))


def streamline_ad_config(config):
    if getattr(config.pipeline.datamanager, "num_processes", None):
        config.pipeline.datamanager.num_processes = 0
    config.pipeline.model.eval_num_rays_per_chunk = 2**17
    if getattr(config.pipeline.datamanager.dataparser, "add_missing_points", None):
        config.pipeline.datamanager.dataparser.add_missing_points = False
    return config


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[RenderRadarFromCameraPath, tyro.conf.subcommand(name="camera-path")],
        Annotated[RadarInterpolatedRender, tyro.conf.subcommand(name="interpolate")],
        Annotated[RadarDatasetRender, tyro.conf.subcommand(name="dataset")],
        Annotated[RadarPoseShiftRender, tyro.conf.subcommand(name="sensor-pose-shift")],
        Annotated[RadarActorRemovalRender, tyro.conf.subcommand(name="actor-removal")],
        Annotated[FullSensorSetRender, tyro.conf.subcommand(name="full-sensor-set")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
