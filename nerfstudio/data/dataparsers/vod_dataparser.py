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

"""Data parser for the View-of-Delft dataset"""
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Type

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix



import numpy as np
import numpy.typing as npt
import torch
from pyquaternion import Quaternion
from torch import Tensor
from typing_extensions import Literal


from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType, transform_points
from nerfstudio.cameras.radars import RADAR_AZIMUTH_RAY_DIVERGENCE, RADAR_ELEVATION_RAY_DIVERGENCE, Radars, RadarType
from nerfstudio.data.dataparsers.ad_dataparser import (
    OPENCV_TO_NERFSTUDIO,
    ADDataParser,
    ADDataParserConfig,
)
from nerfstudio.data.utils.lidar_elevation_mappings import VELODYNE_HDL64ES3_ELEVATION_MAPPING
from nerfstudio.utils import poses as pose_utils

HORIZONTAL_BEAM_DIVERGENCE = 3.0e-3  # radians, or meters at a distance of 1m
VERTICAL_BEAM_DIVERGENCE = 1.5e-3  # radians, or meters at a distance of 1m
MAX_INTENSITY_VALUE = 255.0
VOD_IMAGE_WIDTH = 1936
VOD_IMAGE_HEIGHT = 1216

RADAR_AZIMUTH_RAY_DIVERGENCE = 0.02
RADAR_ELEVATION_RAY_DIVERGENCE = 0.02
RADAR_FOV = [[-1.0, 1.0], [-0.39, 0.49]]

ALLOWED_CATEGORIES = {
    "Car",
    "Cyclist",
    "Moped or scooter",
    "Motor",
    "Truck",
    "Other ride",
    "Other vehicle",
}  # skip Rider, unused bicycle, bicycle rack, human depiction, uncertain ride
SYMMETRIC_CATEGORIES = ALLOWED_CATEGORIES

DEFORMABLE_CATEGORIES = {
    "Pedestrian",
}

DATA_FREQUENCY = 10.0  # 10 Hz
LIDAR_ROTATION_TIME = 1.0 / DATA_FREQUENCY  # 0.1 s
VOD_ELEVATION_MAPPING = {"Velodyne64": VELODYNE_HDL64ES3_ELEVATION_MAPPING}
VOD_AZIMUT_RESOLUTION = {"Velodyne64": 0.1728}

FRAME_STR_LEN = 5
SEQUENCE_TO_FRAME_MAP = {"00": range(100, 400), "01": range(600, 900),
                         "02": range(1400, 1700), "03": range(1850, 2150),
                         "04": range(2220, 2520), "05": range(2532, 2798),
                         "06": range(2900, 3200), "07": range(3277, 3575),
                         "08": range(3575, 3610), "09": range(3650, 3950),
                         "10": range(4050, 4350), "11": range(4387, 4652),
                         "12": range(4660, 4960), "13": range(6334, 6571),
                         "14": range(6571, 6759), "15": range(6800, 7100),
                         "16": range(7600, 7900), "17": range(7900, 8198),
                         "18": range(8198, 8481), "19": range(8482, 8749),
                         "20": range(8749, 9049), "21": range(9100, 9400),
                         "22": range(9518, 9776), "23": range(9776, 9930)}



@dataclass
class VodDataParserConfig(ADDataParserConfig):
    """VoD dataset config.
    VoD (View of Delft) (https://tudelft-iv.github.io/view-of-delft-dataset/) is an autonomous
    driving dataset containing 24 sequences of varying lengths. These sequences include 3+1D Radar,
    Lidar, and Camera data.
    """

    _target: Type = field(default_factory=lambda: Vod)
    """target class to instantiate"""
    sequence: str = "19"
    """Name of the scene."""
    data: Path = Path("data/vod")
    """Path to VoD dataset."""
    split: Literal["training"] = "training"  # we do not have labels for testing set...
    """Which split to use."""
    cameras: Tuple[Literal["front", "none"], ...] = ("front",)
    """Which cameras to use."""
    lidars: Tuple[Literal["velodyne", "none"], ...] = ("velodyne",)
    """Which lidars to use."""
    radars: Tuple[Literal["front", "none"], ...] = ("front",)
    """Which radars to use."""
    annotation_interval: float = 0.1
    """Interval between annotations in seconds."""
    allow_per_point_times: bool = False
    """Whether to allow per-point timestamps."""
    compute_sensor_velocities: bool = False
    """Whether to compute sensor velocities."""
    min_lidar_dist: Tuple[float, ...] = (2.0, 1.6, 2.0) #??
    """Minimum distance of lidar points."""
    lidar_elevation_mapping: Dict[str, Dict] = field(default_factory=lambda: VOD_ELEVATION_MAPPING)
    """Elevation mapping for each lidar."""
    add_missing_points: bool = True
    """Add missing points to lidar point clouds."""
    lidar_azimuth_resolution: Dict[str, float] = field(default_factory=lambda: VOD_AZIMUT_RESOLUTION)
    """Azimuth resolution for each lidar."""
    use_sensor_timestamps: bool = False
    """Whether to use sensor timestamps."""
    include_deformable_actors: bool = True
    """Whether to include deformable actors in the loaded trajectories (like pedestrians)."""


@dataclass
class Vod(ADDataParser):
    """View of Delft DatasetParser"""

    config: VodDataParserConfig

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        """Returns camera info and image filenames."""

        filenames = self.camera_files
        idxs = np.zeros(len(self.camera_files))
        idxs = torch.tensor(np.zeros(len(self.camera_files), dtype=int)).unsqueeze(-1)
        times = torch.tensor(self.camera_times, dtype=torch.float64)

        poses = torch.from_numpy(self.camera_to_odom.copy()).double()
        for frame in range(len(filenames)):
            poses[frame, :3, :3] = poses[frame, :3, :3] @ torch.from_numpy(OPENCV_TO_NERFSTUDIO).double()

        fx = self.calibs["P0"][0, 0].copy()
        fy = self.calibs["P0"][1, 1].copy()
        cx = self.calibs["P0"][0, 2].copy()
        cy = self.calibs["P0"][1, 2].copy()

        cameras = Cameras(
            fx=torch.tensor(fx, dtype=torch.float32),
            fy=torch.tensor(fy, dtype=torch.float32),
            cx=torch.tensor(cx, dtype=torch.float32),
            cy=torch.tensor(cy, dtype=torch.float32),
            height=VOD_IMAGE_HEIGHT,
            width=VOD_IMAGE_WIDTH,
            camera_to_worlds=poses[:, :3, :4].float(),
            camera_type=CameraType.PERSPECTIVE,
            times=times,
            metadata={"sensor_idxs": idxs},
        )
        return cameras, filenames

    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        """Returns lidar info and loaded point clouds."""
        lidar_filenames = self.lidar_files
        times = torch.tensor(self.camera_times, dtype=torch.float64)
        idxs = torch.tensor(np.zeros(len(self.camera_files), dtype=int)).unsqueeze(-1)
        poses = torch.from_numpy(self.camera_to_odom @ self.lidar_to_cam).float()

        lidars = Lidars(
            lidar_to_worlds=poses[:, :3, :4],
            lidar_type=LidarType.VELODYNE64E,
            times=times,
            assume_ego_compensated=True,
            metadata={"sensor_idxs": idxs},
            horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
            vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
        )

        # return lidars, lidar_filenames
        return lidars, lidar_filenames

    def _read_lidars(self, lidars: Lidars, filepaths: List[Path]) -> List[torch.Tensor]:
        lidar_filenames = filepaths
        poses = lidars.lidar_to_worlds
        times = lidars.times.squeeze(1)

        point_clouds = []

        for frame in range(len(lidar_filenames)):
            pc = self.lidar_scans[frame]
            xyz = pc[:, :3]  # N x 3
            intensity = pc[:, 3] / MAX_INTENSITY_VALUE  # N,
            t = get_mock_timestamps(xyz)  # N, relative timestamps
            pc = np.hstack((xyz, intensity[:, None], t[:, None]))
            point_clouds.append(torch.from_numpy(pc).float())

        pcs = []
        if self.config.add_missing_points:
            # remove ego motion compensation
            missing_points = []
            for point_cloud, l2w, time in zip(point_clouds, poses, times):
                pc = point_cloud.clone()
                # absolute time
                pc[:, 4] = pc[:, 4] + time
                # project to world frame
                pc[..., :3] = transform_points(pc[..., :3], l2w.unsqueeze(0).to(pc))
                # remove ego motion compensation
                pc, interpolated_poses = self._remove_ego_motion_compensation(pc.float(), poses, times)

                # Add channel info
                pc = self.add_channel_info(pc, dim=5, lidar_name="Velodyne64")

                # reset time
                pc[:, 4] = point_cloud[:, 4].clone()
                # transform to common lidar frame again
                interpolated_poses = torch.matmul(
                    pose_utils.inverse(l2w.unsqueeze(0)).float(), pose_utils.to4x4(interpolated_poses).float()
                )
                pcs.append(pc)
                # move channel from index 5 to 3
                pc = pc[..., [0, 1, 2, 5, 3, 4]]
                # add missing points
                missing_points.append(self._get_missing_points(pc, interpolated_poses, "Velodyne64", dist_cutoff=0.05)[..., [0, 1, 2, 5, 4, 3]])

            # add missing points to point clouds
            point_clouds = [torch.cat([pc, missing], dim=0) for pc, missing in zip(pcs, missing_points)]

        return point_clouds

    def _get_radars(self):
        """Returns a list of radars."""
        radar_filenames = self.radar_files
        times = torch.tensor(self.camera_times, dtype=torch.float64)
        idxs = torch.tensor(np.zeros(len(self.camera_files), dtype=int)).unsqueeze(-1)
        poses = torch.from_numpy(self.camera_to_odom @ self.radar_to_cam).float()

        radars = Radars(
            radar_to_worlds=poses[:, :3, :4],
            radar_type=RadarType.ZFFRGEN214D,
            assume_ego_compensated=True,
            times=times,
            metadata={"sensor_idxs": idxs},
            radar_azimuth_ray_divergence=RADAR_AZIMUTH_RAY_DIVERGENCE,
            radar_elevation_ray_divergence=RADAR_ELEVATION_RAY_DIVERGENCE,
            min_azimuth=RADAR_FOV[0][0],
            max_azimuth=RADAR_FOV[0][1],
            min_elevation=RADAR_FOV[1][0],
            max_elevation=RADAR_FOV[1][1]
        )

        return radars, radar_filenames

    def _read_radars(self, radars: Radars, filenames: List[Path]) -> List[Tensor]:
        """Reads radar point clouds from the given filenames."""

        assert len(filenames) == len(radars), "Number of vod radar files does not match number of radars."

        radar_pcs = []
        files = np.array(self.radar_files)
        for filename in filenames:
            pc = self.radar_scans[files == filename][0]
            radar_pcs.append(torch.from_numpy(pc).float())

        return radar_pcs

    def _get_actor_trajectories(self) -> List[Dict]:
        """Returns a list of actor trajectories.

        Each trajectory is a dictionary with the following keys:
            - poses: the poses of the actor (float32)
            - timestamps: the timestamps of the actor (float64)
            - dims: the dimensions of the actor, wlh order (float32)
            - label: the label of the actor (str)
            - stationary: whether the actor is stationary (bool)
            - symmetric: whether the actor is expected to be symmetric (bool)
            - deformable: whether the actor is expected to be deformable (e.g. pedestrian)
        """

        if self.config.include_deformable_actors:
            allowed_cats = ALLOWED_CATEGORIES.union(DEFORMABLE_CATEGORIES)
        else:
            allowed_cats = ALLOWED_CATEGORIES

        traj_list = defaultdict(list)

        for frame_id in self.frame_ids:

            anno_file = self.config.data / "lidar" / self.config.split / "label_2" / f"{str(frame_id).zfill(FRAME_STR_LEN)}.txt"

            # read annotations
            try:
                with open(anno_file, "r") as f:
                    lines = f.readlines()
            except:
                continue

            # loop over all annotations to create per agent trajectories
            for line in lines:
                line = line.strip()
                label, track_id, _, _, _, _, _, _, height, width, length, x, y, z, rotation_y, _ = line.split(" ")

                # remove all actors that are not in the allowed categories
                if label not in allowed_cats:
                    continue

                frame = int(frame_id - self.frame_ids[0])
                track_id = int(track_id)

                height = float(height)
                width = float(width)
                length = float(length)

                # defined in the camera coordinate system of the ego-vehicle
                x = float(x)
                y = float(y) - height / 2.0  # center of the object is at the bottom
                z = float(z)
                rotation_y = float(rotation_y)

                traj_list[track_id].append(
                    {
                        "frame": frame,
                        "label": label,
                        "height": height,
                        "width": width,
                        "length": length,
                        "x": x,
                        "y": y,
                        "z": z,
                        "rotation_y": rotation_y,
                    }
                )

        trajs = []
        for track_id, track in traj_list.items():
            poses, wlh, timestamps = [], [], []
            if len(track) < 2:
                continue  # skip if there is only one frame

            label = track[0]["label"]
            deformable = label in DEFORMABLE_CATEGORIES
            symmetric = label in SYMMETRIC_CATEGORIES

            #timestamps = self.camera_times

            # note that the boxes are in the camera coordinate system of the ego-vehicle
            for box in sorted(track, key=lambda x: x["frame"]):
                # cam to world
                cam2world = self.camera_to_odom[box["frame"], ...]
                obj_pose_cam = np.eye(4)
                obj_pose_cam[:3, 3] = np.array([box["x"], box["y"], box["z"]])
                obj_pose_cam[:3, :3] = Quaternion(axis=[0, 1, 0], angle=np.pi / 2 + box["rotation_y"]).rotation_matrix
                obj_pose_world = cam2world @ obj_pose_cam
                poses.append(obj_pose_world)
                wlh.append(np.array([box["width"], box["length"], box["height"]]))
                timestamps.append(self.camera_times[box["frame"]])

            poses = np.array(poses)
            # dynamic if we move more that 1m in any direction
            dynamic = np.any(np.std(poses[:, :3, 3], axis=0) > 0.5)
            # we skip all stationary objects
            if not dynamic:
                continue

            trajs.append(
                {
                    "poses": torch.tensor(poses, dtype=torch.float32),
                    "timestamps": torch.tensor(timestamps, dtype=torch.float64),
                    "dims": torch.tensor(np.median(wlh, axis=0), dtype=torch.float32),
                    "label": label,
                    "stationary": not dynamic,
                    "symmetric": symmetric,
                    "deformable": deformable,
                }
            )

        return trajs

    def add_channel_info(self, point_cloud: torch.Tensor, dim: int = -1, lidar_name: str = "") -> torch.Tensor:
        """Infer channel id from point cloud, and add it to the point cloud.

        Args:
            point_cloud: Point cloud to add channel id to (in sensor frame). Shape: [num_points, 3+x] x,y,z (timestamp, intensity, etc.)

        Returns:
            Point cloud with channel id. Shape: [num_points, 3+x+1] x,y,z (timestamp, intensity, etc.), channel_id
            channel_id is added to dim
        """
        # these are limits where channels are equally spaced
        ELEV_HIGH_IDX = 63

        dist = torch.norm(point_cloud[:, :3], dim=-1)
        elevation = torch.arcsin(point_cloud[:, 2] / dist)
        elevation = torch.rad2deg(elevation)

        middle_elev = elevation#[middle_elev_mask]

        histc, bin_edges = torch.histogram(middle_elev, bins=2000)

        # channels should be equally spaced
        expected_channel_edges = (bin_edges[-1] - bin_edges[0]) / 64 * torch.arange(65) + bin_edges[0]

        res = (
            self.config.lidar_elevation_mapping[lidar_name][ELEV_HIGH_IDX]
            - self.config.lidar_elevation_mapping[lidar_name][ELEV_HIGH_IDX - 1]
        )

        # find consecutive empty bins in histogram
        empty_bins = []
        empty_bin = []
        empty_bins_edges = []
        for i in range(len(histc)):
            if histc[i] == 0:
                empty_bin.append(i)
            else:
                if len(empty_bin) > 0:
                    empty_bins.append(empty_bin)
                    empty_bins_edges.append((bin_edges[empty_bin[0]], bin_edges[empty_bin[-1] + 1]))
                    empty_bin = []

        # find channel edges, use first expected for init
        found_channel_edges = [expected_channel_edges[0].tolist()]
        empty_bins_edges = torch.tensor(empty_bins_edges)
        for i, edge in enumerate(expected_channel_edges[1:-1]):
            found_edge = False
            for empty_bin in empty_bins_edges:
                # if edge is in empty bin, keep the edge as is
                if edge > empty_bin[0] and edge < empty_bin[1]:
                    found_channel_edges.append(edge.tolist())
                    found_edge = True
                    break
            if found_edge:
                continue
            if empty_bins_edges.numel() == 0:
                continue
            else:
                distances = torch.abs(edge - empty_bins_edges)
                min_dist_idx = distances.argmin()
                if distances.flatten()[min_dist_idx] < 0.03:
                    found_channel_edges.append(empty_bins_edges.flatten()[min_dist_idx].tolist())
                    continue

        found_channel_edges.append(expected_channel_edges[-1].tolist())
        found_channel_edges = torch.tensor(found_channel_edges)

        if len(found_channel_edges) < len(expected_channel_edges):
            # we have missing channels, interpolate edges
            while (num_missing_edges := len(expected_channel_edges) - len(found_channel_edges)) > 0:
                distances = found_channel_edges.diff().abs()
                max_dist_idx = distances.argmax()
                num_edges_to_insert = max((distances[max_dist_idx] / res).round().int() - 1, 1)
                num_edges_to_insert = min(num_missing_edges, num_edges_to_insert)
                new_edges = torch.linspace(
                    found_channel_edges[max_dist_idx], found_channel_edges[max_dist_idx + 1], num_edges_to_insert + 2
                )[1:-1]
                found_channel_edges = torch.cat(
                    [found_channel_edges[: max_dist_idx + 1], new_edges, found_channel_edges[max_dist_idx + 1 :]]
                )  # insert new edges

        found_channel_edges, _ = torch.sort(found_channel_edges, descending=True)
        channel_id = torch.full((point_cloud.shape[0], 1), -1, device=point_cloud.device)

        # assign channel id
        for i in range(len(self.config.lidar_elevation_mapping[lidar_name])):
            elevation_mask = (elevation >= found_channel_edges[i + 1]) & (elevation < found_channel_edges[i])
            channel_id[elevation_mask] = i

        point_cloud = torch.cat([point_cloud[:, :dim], channel_id, point_cloud[:, dim:]], dim=-1)
        return point_cloud

    def _generate_dataparser_outputs(self, split="train"):
        # load the ego_poses
        kitti_locations = KittiLocations(root_dir=str(self.config.data))

        frames = SEQUENCE_TO_FRAME_MAP[self.config.sequence]
        camera_files = []
        lidar_files = []
        radar_files = []
        camera_times = np.zeros((len(frames)))
        lidar_scans = np.zeros((len(frames)), dtype=np.ndarray)
        radar_scans = np.zeros((len(frames)), dtype=np.ndarray)
        camera_to_odom = np.zeros((len(frames), 4, 4))
        lidar_to_cam = np.zeros((len(frames), 4, 4))
        radar_to_cam = np.zeros((len(frames), 4, 4))
        camera_matrix = np.empty((3, 4))
        calibs = {}
        for frame in frames:
            camera_times[frame - frames[0]] = (frame - frames[0]) / DATA_FREQUENCY
            frame_str = str(frame).zfill(FRAME_STR_LEN)
            frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_str)
            transforms = FrameTransformMatrix(frame_data)
            camera_to_odom[frame - frames[0]] = transforms.t_odom_camera
            lidar_to_cam[frame - frames[0]] = transforms.t_camera_lidar
            radar_to_cam[frame - frames[0]] = transforms.t_camera_radar
            camera_files.append(Path(kitti_locations.camera_dir) / f"{frame_str}.jpg")
            lidar_files.append(Path(kitti_locations.lidar_dir) / f"{frame_str}.bin")
            radar_files.append(Path(kitti_locations.radar_dir) / f"{frame_str}.bin")
            lidar_scans[frame - frames[0]] = frame_data.lidar_data
            radar_scans[frame - frames[0]] = frame_data.radar_data
            if frame == frames[0]:
                camera_matrix = transforms.camera_projection_matrix
                calibs = get_calib(Path(frame_data.kitti_locations.lidar_calib_dir) / f"{frame_str}.txt")

        self.camera_to_odom = camera_to_odom
        self.lidar_to_cam = lidar_to_cam
        self.radar_to_cam = radar_to_cam
        self.ego_poses = np.linalg.inv(self.camera_to_odom[0]) @ self.camera_to_odom
        self.camera_matrix = camera_matrix
        self.camera_times = camera_times
        self.calibs = calibs
        self.camera_files = camera_files
        self.lidar_files = lidar_files
        self.lidar_scans = lidar_scans
        self.radar_files = radar_files
        self.radar_scans = radar_scans
        self.frame_ids = frames

        out = super()._generate_dataparser_outputs(split=split)
        del self.calibs
        del self.camera_files
        del self.camera_matrix
        del self.ego_poses
        del self.camera_to_odom
        del self.lidar_to_cam
        del self.radar_to_cam
        del self.camera_times
        del self.lidar_files
        del self.lidar_scans
        del self.radar_files
        del self.radar_scans
        del self.frame_ids
        return out

    def _setup_sensor_timestamps(self) -> None:
        self.timestamp_per_sensor = {}
        for sensor in self.config.cameras + self.config.lidars + ("oxts",):
            timestamp_file = (
                self.config.data
                / self.config.split
                / sensor
                / "timestamps"
                / f"{self.config.sequence}"
                / "timestamps.txt"
            )
            assert timestamp_file.exists(), f"Trying to use sensor timestamps but file {timestamp_file} does not exist."
            with open(timestamp_file, "r") as f:
                lines = f.readlines()
            # parse the timestamps: 2011-09-26 13:13:32.322364840
            timestamps = []
            for line in lines:
                line = line.strip()[:-3]  # remove nanosecond precision
                dt = datetime.strptime(line, "%Y-%m-%d %H:%M:%S.%f")
                timestamp = dt.timestamp()
                timestamps.append(timestamp)

            self.timestamp_per_sensor[sensor] = np.array(timestamps)


    def _get_linspaced_indices(self, sensor_idxs: Tensor) -> Tuple[Tensor, Tensor]:
        # if we are using all the samples, i.e., optimizing poses, we can use the same for eval
        if self.config.train_split_fraction == 1.0:
            return torch.arange(sensor_idxs.numel(), dtype=torch.int64), torch.arange(
                sensor_idxs.numel(), dtype=torch.int64
            )
        else:
            return super()._get_linspaced_indices(sensor_idxs)




def get_calib(calib_path: Path) -> Dict[str, npt.NDArray[np.float32]]:
    with open(str(calib_path), "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("P0:"):
            P0 = np.array(line.split(" ")[1:], dtype=np.float32).reshape(3, 4)
        elif line.startswith("P1:"):
            P1 = np.array(line.split(" ")[1:], dtype=np.float32).reshape(3, 4)
        elif line.startswith("P2:"):
            P2 = np.array(line.split(" ")[1:], dtype=np.float32).reshape(3, 4)
        elif line.startswith("P3:"):
            P3 = np.array(line.split(" ")[1:], dtype=np.float32).reshape(3, 4)
        elif line.startswith("R0_rect"):
            R_rect = np.array(line.split(" ")[1:], dtype=np.float32).reshape(3, 3)
        elif line.startswith("Tr_velo_to_cam"):
            Tr_velo_to_cam = np.array(line.split(" ")[1:], dtype=np.float32).reshape(3, 4)
        elif line.startswith("Tr_imu_to_velo"):
            continue
        else:
            raise ValueError(f"Unknown calibration line: {line}")
    return {
        "P0": P0,  # type: ignore
        "P1": P1,  # type: ignore
        "P2": P2,  # type: ignore
        "P3": P3,  # type: ignore
        "R_rect": R_rect,  # type: ignore
        "Tr_velo_to_cam": Tr_velo_to_cam,  # type: ignore
    }


def get_mock_timestamps(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Get mock relative timestamps for the velodyne points."""
    # the velodyne has x forward, y left, z up and the sweep is split behind the car.
    # it is also rotating counter-clockwise, meaning that the angles close to -pi are the
    # first ones in the sweep and the ones close to pi are the last ones in the sweep.
    angles = np.arctan2(points[:, 1], points[:, 0])  # N, [-pi, pi]
    angles += np.pi  # N, [0, 2pi]
    # see how much of the rotation have finished
    fraction_of_rotation = angles / (2 * np.pi)  # N, [0, 1]
    # get the pseudo timestamps based on the total rotation time
    timestamps = fraction_of_rotation * LIDAR_ROTATION_TIME
    return timestamps