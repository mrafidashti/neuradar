# Copyright 2025 the authors of NeuRadar and contributors.
# Copyright 2025 the authors of NeuRAD and contributors.
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
"""
Radar Models
"""
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Union

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import Parameter

import nerfstudio.utils.math
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.lidars import transform_points, transform_points_pairwise
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.misc import strtobool
from nerfstudio.utils.tensor_dataclass import TensorDataclass

TORCH_DEVICE = Union[torch.device, str]  # pylint: disable=invalid-name

RADAR_AZIMUTH_RAY_DIVERGENCE = 0.0625  # theta in spherical coords
RADAR_ELEVATION_RAY_DIVERGENCE = 0.0625  # phi in spherical coords

MIN_AZIMUTH = -0.5
MAX_AZIMUTH = 0.5
MIN_ELEVATION = -0.5
MAX_ELEVATION = 0.5


class RadarType(Enum):
    """Supported Radar Types."""

    ZFFRGEN214D = auto()
    ContiARS40821 = auto()
    ContiFLR2 = auto()


RADAR_MODEL_TO_TYPE = {
    "ZFFRGEN214D": RadarType.ZFFRGEN214D,
    "ContiARS40821": RadarType.ContiARS40821,
    "ContiFLR2": RadarType.ContiFLR2,
}


@dataclass(init=False)
class Radars(TensorDataclass):
    """Dataparser outputs for the radar dataset and the ray generator.

    If a single value is provided, it is broadcasted to all radars.

    Args:
        radar_to_worlds: Radar to world matrices. Tensor of per-image r2w matrices, in [R | t] format
        radar_type: Type of radar sensor. This will be an int corresponding to the RadarType enum.
        assume_ego_compensated: Whether or not to assume that points are ego-compensated.
        times: Timestamps for each radar
        n_points: Number of points in each radar
        metadata: Additional metadata or data needed for interpolation, will mimic shape of the radars
            and will be broadcasted to the rays generated from any derivative RaySamples we create with this
    """

    radar_to_worlds: Float[Tensor, "*num_radars 3 4"]
    radar_type: Float[Tensor, "*num_radars 1"]
    times: Optional[Int[Tensor, "*num_radars 1"]]
    metadata: Optional[Dict]
    radar_azimuth_ray_divergence: Float[Tensor, "*num_radars 1"]
    radar_elevation_ray_divergence: Float[Tensor, "*num_radars 1"]
    min_azimuth: Float[Tensor, "*num_radars 1"]
    max_azimuth: Float[Tensor, "*num_radars 1"]
    min_elevation: Float[Tensor, "*num_radars 1"]
    max_elevation: Float[Tensor, "*num_radars 1"]

    def __init__(
        self,
        radar_to_worlds: Float[Tensor, "*batch_r2ws 3 4"],
        radar_type: Optional[
            Union[
                Int[Tensor, "*batch_radar_types 1"],
                int,
                List[RadarType],
                RadarType,
            ]
        ] = RadarType.ZFFRGEN214D,
        assume_ego_compensated: bool = True,
        times: Optional[Float[Tensor, "*num_radars"]] = None,
        metadata: Optional[Dict] = None,
        radar_azimuth_ray_divergence: Optional[Union[Float, Float[Tensor, "*num_radars 1"]]] = None,
        radar_elevation_ray_divergence: Optional[Union[Float, Float[Tensor, "*num_radars 1"]]] = None,
        min_azimuth: Optional[Union[Float, Float[Tensor, "*num_radars 1"]]] = None,
        max_azimuth: Optional[Union[Float, Float[Tensor, "*num_radars 1"]]] = None,
        min_elevation: Optional[Union[Float, Float[Tensor, "*num_radars 1"]]] = None,
        max_elevation: Optional[Union[Float, Float[Tensor, "*num_radars 1"]]] = None,
        valid_radar_distance_threshold: float = 300,  # do we need this?
    ) -> None:
        """Initializes the Lidars object.

        Note on Input Tensor Dimensions: All of these tensors have items of dimensions Shaped[Tensor, "3 4"]
        (in the case of the c2w matrices), Shaped[Tensor, "6"] (in the case of distortion params), or
        Shaped[Tensor, "1"] (in the case of the rest of the elements). The dimensions before that are
        considered the batch dimension of that tensor (batch_c2ws, batch_fxs, etc.). We will broadcast
        all the tensors to be the same batch dimension. This means you can use any combination of the
        input types in the function signature and it won't break. Your batch size for all tensors
        must be broadcastable to the same size, and the resulting number of batch dimensions will be
        the batch dimension with the largest number of dimensions.
        """

        # This will notify the tensordataclass that we have a field with more than 1 dimension
        self._field_custom_dimensions = {"radar_to_worlds": 2}

        self.radar_to_worlds = radar_to_worlds

        # @dataclass's post_init will take care of broadcasting
        self.radar_type = self._init_get_radar_type(radar_type)  # type: ignore
        self.times = self._init_get_times(times)

        self.metadata = metadata

        self.radar_azimuth_ray_divergence = self._init_get_ray_divergence(
            radar_azimuth_ray_divergence, RADAR_AZIMUTH_RAY_DIVERGENCE
        )

        self.radar_elevation_ray_divergence = self._init_get_ray_divergence(
            radar_elevation_ray_divergence, RADAR_ELEVATION_RAY_DIVERGENCE
        )

        self.min_azimuth = self._init_get_ray_divergence(min_azimuth, MIN_AZIMUTH)
        self.max_azimuth = self._init_get_ray_divergence(max_azimuth, MAX_AZIMUTH)
        self.min_elevation = self._init_get_ray_divergence(min_elevation, MIN_ELEVATION)
        self.max_elevation = self._init_get_ray_divergence(max_elevation, MAX_ELEVATION)

        self.__post_init__()  # This will do the dataclass post_init and broadcast all the tensors

        self._use_nerfacc = strtobool(os.environ.get("INTERSECT_WITH_NERFACC", "TRUE"))
        self.assume_ego_compensated = assume_ego_compensated

        self.valid_radar_distance_threshold = valid_radar_distance_threshold

    def _init_get_radar_type(
        self,
        radar_type: Union[
            Int[Tensor, "*batch_radar_types 1"], Int[Tensor, "*batch_radar_types"], int, List[RadarType], RadarType
        ],
    ) -> Int[Tensor, "*num_radars 1"]:
        """
        Parses the __init__() argument lidar_type

        Lidar Type Calculation:
        If LidarType, convert to int and then to tensor, then broadcast to all lidars
        If List of LidarTypes, convert to ints and then to tensor, then broadcast to all lidars
        If int, first go to tensor and then broadcast to all lidars
        If tensor, broadcast to all lidars

        Args:
            radar_type: radar_type argument from __init__()
        """
        if isinstance(radar_type, RadarType):
            radar_type = torch.tensor([radar_type.value], device=self.device)
        elif isinstance(radar_type, List) and isinstance(radar_type[0], RadarType):
            radar_type = torch.tensor([[c.value] for c in radar_type], device=self.device)
        elif isinstance(radar_type, int):
            radar_type = torch.tensor([radar_type], device=self.device)
        elif isinstance(radar_type, torch.Tensor):
            assert not torch.is_floating_point(
                radar_type
            ), f"radar_type tensor must be of type int, not: {radar_type.dtype}"
            radar_type = radar_type.to(self.device)
            if radar_type.ndim == 0 or radar_type.shape[-1] != 1:
                radar_type = radar_type.unsqueeze(-1)
        else:
            raise ValueError(
                'Invalid radar_type. Must be RadarType, List[RadarType], int, or torch.Tensor["num_radars"]. \
                    Received: '
                + str(type(radar_type))
            )
        return radar_type

    def _init_get_times(self, times: Union[None, torch.Tensor]) -> Union[None, torch.Tensor]:
        if times is None:
            times = None
        elif isinstance(times, torch.Tensor):
            if times.ndim == 0 or times.shape[-1] != 1:
                times = times.unsqueeze(-1).to(self.device)
        else:
            raise ValueError(f"times must be None or a tensor, got {type(times)}")

        return times

    def _init_get_ray_divergence(
        self,
        ray_divergence: Union[None, Float, Float[Tensor, "*num_radars 1"]],
        default: float,
    ) -> Float[Tensor, "*num_radars 1"]:
        if ray_divergence is None:
            ray_divergence = torch.ones_like(self.radar_type, device=self.device) * default
        elif isinstance(ray_divergence, float):
            ray_divergence = torch.ones_like(self.radar_type, device=self.device) * ray_divergence
        elif isinstance(ray_divergence, torch.Tensor):
            if ray_divergence.ndim == 0 or ray_divergence.shape[-1] != 1:
                ray_divergence = ray_divergence.unsqueeze(-1)
        else:
            raise ValueError(f"Ray divergence must be None, float, or tensor, got {type(ray_divergence)}")

        return ray_divergence

    @property
    def device(self) -> TORCH_DEVICE:
        """Returns the device that the radar is on."""
        return self.radar_to_worlds.device

    def generate_rays(
        self,
        scan_indices: Optional[Int] = None,
        keep_shape: Optional[bool] = None,
        aabb_box: Optional[SceneBox] = None,
    ) -> RayBundle:
        # If zero dimensional, we need to unsqueeze to get a batch dimension and then squeeze later
        if not self.shape:
            radars = self.reshape((1,))
        else:
            radars = self

        ray_bundle = radars._generate_rays_from_fov(scan_indices=scan_indices)

        # If we have mandated that we don't keep the shape, then we flatten
        if keep_shape is False:
            raybundle = raybundle.flatten()

        if aabb_box:
            with torch.no_grad():
                tensor_aabb = Parameter(aabb_box.aabb.flatten(), requires_grad=False)

                rays_o = raybundle.origins.contiguous()
                rays_d = raybundle.directions.contiguous()

                tensor_aabb = tensor_aabb.to(rays_o.device)
                shape = rays_o.shape

                rays_o = rays_o.reshape((-1, 3))
                rays_d = rays_d.reshape((-1, 3))

                t_min, t_max = nerfstudio.utils.math.intersect_aabb(rays_o, rays_d, tensor_aabb)

                t_min = t_min.reshape([shape[0], shape[1], 1])
                t_max = t_max.reshape([shape[0], shape[1], 1])

                raybundle.nears = t_min
                raybundle.fars = t_max

        # See Georg's comment in lidars.py
        return ray_bundle

    def _generate_rays_from_fov(
        self,
        scan_indices: Optional[Int[Tensor, "n_scans"]] = None,
    ) -> RayBundle:
        # Make sure we're on the right devices
        scan_indices = scan_indices.to(self.device)

        directions_spher = torch.empty((0, 2), device=self.device)
        scan_indices_for_rays = torch.empty(0, dtype=torch.int64, device=self.device)
        for index in scan_indices:
            # 2d matrix of azimuths and evalations <--> rays for this scan
            azimuths = torch.arange(
                self.min_azimuth[index, 0],
                self.max_azimuth[index, 0],
                self.radar_azimuth_ray_divergence[index, 0],
                device=self.device,
            )
            elevations = torch.arange(
                self.min_elevation[index, 0],
                self.max_elevation[index, 0],
                self.radar_elevation_ray_divergence[index, 0],
                device=self.device,
            )
            grid_azimuths, grid_elevations = torch.meshgrid(
                azimuths, elevations, indexing="ij"
            )  # meshgrid of azimuths and elevations
            directions_spher_scan = torch.stack((grid_azimuths.flatten(), grid_elevations.flatten()), dim=1)  # (N, 2)
            directions_spher = torch.cat((directions_spher, directions_spher_scan))

            # record the associations between the rays and scans
            num_rays_scan = directions_spher_scan.shape[0]
            scan_indices_for_rays = torch.cat(
                (scan_indices_for_rays, torch.full((num_rays_scan,), index, dtype=torch.int64, device=self.device))
            )
        num_rays_shape = scan_indices_for_rays.shape  # (N,)

        # r2w matrix
        r2w = self.radar_to_worlds[scan_indices_for_rays, ...]  # (N,3,4) where N represents number of rays
        assert r2w.shape == num_rays_shape + (3, 4)

        # origins
        origins = r2w[..., :3, 3]  # (N,3)
        assert origins.shape == num_rays_shape + (3,)

        # convert spherical to cartesian, and then obtain directions
        directions = torch.zeros((directions_spher.shape[0], 3), device=self.device)
        directions[:, 0] = torch.cos(directions_spher[:, 1]) * torch.cos(directions_spher[:, 0])
        directions[:, 1] = torch.cos(directions_spher[:, 1]) * torch.sin(directions_spher[:, 0])
        directions[:, 2] = torch.sin(directions_spher[:, 1])
        directions = transform_points_pairwise(directions, r2w)  # (N,3)
        directions, distance = camera_utils.normalize_with_norm(
            directions - origins, -1
        )  # all the values in distance are 1.00; should keep it?
        assert directions.shape == num_rays_shape + (3,)

        # pixel_area
        dx = (
            self.radar_azimuth_ray_divergence[scan_indices_for_rays] / 5
        )  # (N,1); 5 is a hyperparameter, which could be tuned
        dy = self.radar_elevation_ray_divergence[scan_indices_for_rays] / 5
        pixel_area = dx * dy  # (N,1)
        assert pixel_area.shape == num_rays_shape + (1,)

        # metadata
        metadata = (
            self._apply_fn_to_dict(self.metadata, lambda x: x[scan_indices_for_rays])
            if self.metadata is not None
            else None
        )
        if metadata is not None:
            metadata["directions_norm"] = distance.detach()
        else:
            metadata = {"directions_norm": distance.detach()}
        metadata["did_return"] = torch.ones(
            (directions.shape[0], 1), dtype=torch.bool, device=self.device
        ).detach()  # (N,1)
        metadata["directions_spher"] = directions_spher  # (N,2)

        # times
        times = self.times[scan_indices_for_rays] if self.times is not None else None  # (N,1)

        return RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=scan_indices_for_rays.unsqueeze(-1),
            times=times,
            metadata=metadata,  # type: ignore
            fars=torch.ones_like(pixel_area, device=self.device)
            * 1_000_000,  # TODO: is this cheating? (from lidars.py)
        )

    def to_json(self, radar_idx: int, sensor_idx: int, point_cloud: Tensor, max_points: Optional[int] = None) -> Dict:
        """Convert a radar point cloud to a json dictionary.

        Args:
            radar_idx: Index of the radar data to convert.
            point_cloud: Point cloud to convert.
            max_size: Max size to resize the image to if present.

        Returns:
            A JSON representation of the camera
        """
        # convert point cloud to world coordinates and then subsample and write to json
        point_cloud = transform_points(point_cloud, self.radar_to_worlds[radar_idx])
        if max_points is not None and point_cloud.shape[0] > max_points:
            point_cloud = point_cloud[torch.randperm(point_cloud.shape[0])[:max_points]]
        json_ = {
            "sensor_idx": sensor_idx,
            "radar_idx": radar_idx,
            "points": point_cloud[:, :4].tolist(),
            "radar_to_world": self.radar_to_worlds[radar_idx].tolist(),
        }
        return json_
