# Copyright 2025 the authors of NeuRadar and contributors.
# Copyright 2025 the authors of NeuRAD and contributors.
# Copyright 2025 the authors of NeuRadar and contributors.
# Copyright 2024 the authors of NeuRAD and contributors.
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
Lidar and Radar Datamanager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.multiprocessing as mp
from rich.progress import Console
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import TDataset
from nerfstudio.data.datamanagers.image_lidar_datamanager import (
    ImageLidarDataManager,
    ImageLidarDataManagerConfig,
    ImageLidarDataProcessor,
    _cache_images,
    _cache_points,
    lidar_packed_collate,
)
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager, ParallelDataManagerConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datasets.lidar_dataset import LidarDataset
from nerfstudio.data.datasets.radar_dataset import RadarDataset
from nerfstudio.data.pixel_samplers import (
    LidarPointSampler,
    LidarPointSamplerConfig,
    PixelSampler,
    RadarPointSampler,
    RadarPointSamplerConfig,
)
from nerfstudio.data.utils.dataloaders import CacheDataloader, FixedIndicesEvalDataloader, RandIndicesEvalDataloader
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import LidarRayGenerator, RadarRayGenrator

CONSOLE = Console(width=120)


def radar_packed_collate(batch: List[Dict]) -> Dict:
    """Default collate function for the cached dataloader.
    Args:
        batch: Batch of samples from the dataset.
    Returns:
        Collated batch.
    """
    radars = []
    for i, data in enumerate(batch):
        batch[i]["points_per_radar"] = data["radar"].shape[0]
        radars.append(data.pop("radar"))

    new_batch: dict = nerfstudio_collate(batch)
    new_batch["radar"] = torch.cat(radars, dim=0)
    return new_batch


@dataclass
class ImageLidarRadarDataManagerConfig(ImageLidarDataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: ImageLidarRadarDataManager)
    """Target class to instantiate."""
    train_num_radar_scans_per_batch: int = 16
    """Number of radar scans per batch to use in each training iteration."""
    eval_num_radar_scans_per_batch: int = 1
    """Number of radar scans per batch to use in each eval iteration."""


class ImageLidarRadarDataProcessor(ImageLidarDataProcessor):  # type: ignore
    """Parallel dataset batch processor.

    This class is responsible for generating ray bundles from an input dataset
    in parallel python processes.

    Args:
        out_queue: the output queue for storing the processed data
        config: configuration object for the parallel data manager
        dataparser_outputs: outputs from the dataparser
        dataset: input dataset
        pixel_sampler: The pixel sampler for sampling rays
        ray_generator: The ray generator for generating rays
    """

    def __init__(
        self,
        out_queue: Any[mp.Queue],  # type: ignore
        func_queue: Any[mp.Queue],  # type: ignore
        config: ParallelDataManagerConfig,
        dataparser_outputs: DataparserOutputs,
        image_dataset: TDataset,
        pixel_sampler: PixelSampler,
        lidar_dataset: LidarDataset,
        point_sampler: LidarPointSampler,
        radar_point_sampler: RadarPointSampler,
        radar_dataset: RadarDataset,
        cached_images: Dict[str, torch.Tensor],
        cached_points: Dict[str, torch.Tensor],
        cached_radar_points: Dict[str, torch.Tensor],
    ):
        super().__init__(
            out_queue=out_queue,
            func_queue=func_queue,
            config=config,
            dataparser_outputs=dataparser_outputs,
            image_dataset=image_dataset,
            pixel_sampler=pixel_sampler,
            lidar_dataset=lidar_dataset,
            point_sampler=point_sampler,
            cached_images=cached_images,
            cached_points=cached_points,
        )

        self.radar_dataset = radar_dataset
        self.radar_point_sampler = radar_point_sampler
        self.radar_ray_generator = RadarRayGenrator(self.radar_dataset.radars)
        self.cached_radar_points = cached_radar_points

    def get_batch_and_ray_bundle(self):
        img_batch, img_ray_bundle = self.get_image_batch_and_ray_bundle()
        lidar_batch, lidar_ray_bundle = self.get_lidar_batch_and_ray_bundle()
        radar_batch, radar_ray_bundle = self.get_radar_batch_and_ray_bundle()
        return _merge_img_lidar_radar(
            img_ray_bundle,
            img_batch,
            lidar_ray_bundle,
            lidar_batch,
            len(self.image_dataset),
            len(self.lidar_dataset),
            radar_ray_bundle,
            radar_batch,
        )

    def get_radar_batch_and_ray_bundle(self):
        if not len(self.radar_dataset.radars):
            return None, None
        batch, scan_indices = self.radar_point_sampler.sample(self.cached_radar_points)
        batch["radar_scan_indices"] = scan_indices
        ray_bundle: RayBundle = self.radar_ray_generator(scan_indices=scan_indices)
        return batch, ray_bundle


class ImageLidarRadarDataManager(ImageLidarDataManager):
    """This extends the VanillaDataManager to support lidar data.

    Args:
        config: the ImageLidarRadarDataManagerConfig used to instantiate class
    """

    config: ImageLidarRadarDataManagerConfig  # type: ignore[override]
    train_radar_dataset: RadarDataset
    eval_radar_dataset: RadarDataset

    def __init__(
        self,
        config: ImageLidarRadarDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)

    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""
        img_dataset = super().create_train_dataset()
        self.train_radar_dataset = RadarDataset(self.train_dataparser_outputs)
        return img_dataset

    def create_eval_dataset(self) -> InputDataset:
        """Sets up the data loaders for evaluation"""
        img_dataset = super().create_eval_dataset()
        self.eval_radar_dataset = RadarDataset(self.eval_dataparser_outputs)
        return img_dataset

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)  # type: ignore
        self.train_point_sampler = LidarPointSamplerConfig().setup(
            num_rays_per_batch=self.config.train_num_lidar_rays_per_batch
        )
        self.train_lidar_ray_generator = LidarRayGenerator(
            self.train_lidar_dataset.lidars,
        )
        self.train_radar_point_sampler = RadarPointSamplerConfig().setup(
            num_radar_scans_per_batch=self.config.train_num_radar_scans_per_batch
        )
        # Cache jointly to allow memory sharing between processes
        cached_images = _cache_images(self.train_dataset, self.config.max_thread_workers, self.config.collate_fn)
        cached_points = _cache_points(self.train_lidar_dataset, self.config.max_thread_workers, lidar_packed_collate)
        cached_radar_points = _cache_points(
            self.train_radar_dataset, self.config.max_thread_workers, radar_packed_collate
        )
        self.data_queue = mp.Queue(maxsize=self.config.queue_size) if self.use_mp else None
        # Create an individual queue for passing functions to each process
        self.func_queues = [mp.Queue() for _ in range(max(self.config.num_processes, 1))]
        self.data_procs = [
            ImageLidarRadarDataProcessor(
                out_queue=self.data_queue,
                func_queue=func_queue,
                config=self.config,
                dataparser_outputs=self.train_dataparser_outputs,
                image_dataset=self.train_dataset,
                pixel_sampler=self.train_pixel_sampler,
                lidar_dataset=self.train_lidar_dataset,
                point_sampler=self.train_point_sampler,
                radar_point_sampler=self.train_radar_point_sampler,
                radar_dataset=self.train_radar_dataset,
                cached_images=cached_images,
                cached_points=cached_points,
                cached_radar_points=cached_radar_points,
            )
            for func_queue in self.func_queues
        ]
        if self.use_mp:
            for proc in self.data_procs:
                proc.start()
            print("Started processes")

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        super().setup_eval()

        self.eval_radar_dataloader = CacheDataloader(
            self.eval_radar_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=radar_packed_collate,
        )
        self.iter_eval_radar_dataloader = iter(self.eval_radar_dataloader)

        self.eval_radar_point_sampler = RadarPointSamplerConfig().setup(
            num_radar_scans_per_batch=self.config.eval_num_radar_scans_per_batch
        )
        self.eval_radar_ray_generator = RadarRayGenrator(self.eval_radar_dataset.radars.to(self.device))

        self.fixed_indices_eval_radar_dataloader = FixedIndicesEvalDataloader(
            dataset=self.eval_radar_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.rand_indices_eval_radar_dataloader = RandIndicesEvalDataloader(
            dataset=self.eval_radar_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        if len(self.eval_dataset.cameras):
            img_ray_bundle, img_batch = ParallelDataManager.next_eval(self, step)
        else:
            self.eval_count += 1
            img_ray_bundle, img_batch = None, None
        if len(self.eval_lidar_dataset.lidars):
            lidar_batch = next(self.iter_eval_lidar_dataloader)
            lidar_batch = self.eval_point_sampler.sample(lidar_batch)
            lidar_ray_bundle = self.eval_lidar_ray_generator(lidar_batch.pop("indices"), points=lidar_batch["lidar"])
        else:
            lidar_ray_bundle, lidar_batch = None, None
        if len(self.eval_radar_dataset.radars):
            radar_batch = next(self.iter_eval_radar_dataloader)
            radar_batch, scan_indices = self.eval_radar_point_sampler.sample(radar_batch)
            radar_batch["radar_scan_indices"] = scan_indices
            radar_ray_bundle = self.eval_radar_ray_generator(scan_indices=scan_indices)
        else:
            radar_ray_bundle, radar_batch = None, None
        return _merge_img_lidar_radar(
            img_ray_bundle,
            img_batch,
            lidar_ray_bundle,
            lidar_batch,
            len(self.eval_dataset.cameras),
            len(self.eval_lidar_dataset.lidars),
            radar_ray_bundle,
            radar_batch,
        )

    def get_num_train_data(self) -> int:
        """Get the number of training datapoints (images + lidar + radar scans)."""
        return (
            len(self.train_dataset.cameras)
            + len(self.train_lidar_dataset.lidars)
            + len(self.train_radar_dataset.radars)
        )

    def next_eval_radar(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for _, batch in self.rand_indices_eval_radar_dataloader:
            radar_idx = batch["radar_idx"]
            num_points = batch["radar"].shape[0]
            ray_indices = torch.cat(
                [
                    torch.full((num_points, 1), radar_idx, dtype=torch.int64, device=self.device),
                    torch.arange(num_points, device=self.device).view(-1, 1),
                ],
                dim=-1,
            )  # actually, it is useless for generating rays
            batch["indices"] = ray_indices
            scan_index = torch.tensor([radar_idx])
            ray_bundle = self.eval_radar_dataset.radars.generate_rays(scan_indices=scan_index).to(self.device)

            is_radar = torch.ones((len(ray_bundle), 1), dtype=torch.bool, device=self.device)
            ray_bundle.metadata["is_radar"], batch["is_radar"] = is_radar, is_radar
            batch["distance"] = ray_bundle.metadata["directions_norm"]
            batch["did_return"] = ray_bundle.metadata["did_return"]
            return radar_idx, ray_bundle, batch
        raise ValueError("No more eval images")


def _merge_img_lidar_radar(
    img_ray_bundle: Optional[RayBundle],
    img_batch: Optional[Dict],
    lidar_ray_bundle: Optional[RayBundle],
    lidar_batch: Optional[Dict],
    img_dataset_len: int,
    lidar_dataset_len: int,
    radar_ray_bundle: Optional[RayBundle],
    radar_batch: Optional[Dict],
) -> Tuple[RayBundle, Dict]:
    """Helper function for merging img and lidar data."""
    if img_ray_bundle is None and lidar_ray_bundle is None:
        raise ValueError("Need either img or lidar data (or both)")

    # process image
    if img_ray_bundle is not None:
        assert img_batch is not None
        device = img_ray_bundle.origins.device
        img_batch["img_indices"] = img_batch.pop("indices")
        img_batch["is_lidar"] = torch.zeros((len(img_ray_bundle), 1), dtype=torch.bool, device=device)
        img_batch["did_return"] = torch.ones((len(img_ray_bundle), 1), dtype=torch.bool, device=device)
        img_ray_bundle.metadata["is_lidar"] = img_batch["is_lidar"]
        img_ray_bundle.metadata["did_return"] = img_batch["did_return"]
        img_batch["is_radar"] = torch.zeros((len(img_ray_bundle), 1), dtype=torch.bool, device=device)
        img_ray_bundle.metadata["is_radar"] = img_batch["is_radar"]
        img_ray_bundle.metadata["directions_spher"] = torch.zeros((len(img_ray_bundle), 2), device=device)

    # process lidar
    if lidar_ray_bundle is not None:
        assert lidar_batch is not None
        device = lidar_ray_bundle.origins.device
        lidar_batch["is_lidar"] = torch.ones((len(lidar_ray_bundle), 1), dtype=torch.bool, device=device)
        lidar_ray_bundle.metadata["is_lidar"] = lidar_batch["is_lidar"]
        lidar_batch["did_return"] = lidar_ray_bundle.metadata["did_return"]
        lidar_batch["distance"] = lidar_ray_bundle.metadata["directions_norm"]
        lidar_batch["is_radar"] = torch.zeros((len(lidar_ray_bundle), 1), dtype=torch.bool, device=device)
        lidar_ray_bundle.metadata["is_radar"] = lidar_batch["is_radar"]
        lidar_ray_bundle.metadata["directions_spher"] = torch.zeros((len(lidar_ray_bundle), 2), device=device)
        lidar_ray_bundle.camera_indices = lidar_ray_bundle.camera_indices + img_dataset_len

    # process radar
    if radar_ray_bundle is not None:
        assert radar_batch is not None
        device = radar_ray_bundle.origins.device
        radar_batch["is_lidar"] = torch.zeros((len(radar_ray_bundle), 1), dtype=torch.bool, device=device)
        radar_ray_bundle.metadata["is_lidar"] = radar_batch["is_lidar"]
        radar_batch["did_return"] = radar_ray_bundle.metadata["did_return"]
        radar_batch["is_radar"] = torch.ones((len(radar_ray_bundle), 1), dtype=torch.bool, device=device)
        radar_ray_bundle.metadata["is_radar"] = radar_batch["is_radar"]
        radar_batch["radar_indices"] = radar_batch.pop("indices")
        radar_ray_bundle.camera_indices = radar_ray_bundle.camera_indices + lidar_dataset_len + img_dataset_len

    # merge
    if (img_ray_bundle is None or img_batch is None) and (radar_ray_bundle is None or radar_batch is None):
        ray_bundle, batch = lidar_ray_bundle, lidar_batch
    elif (lidar_ray_bundle is None or lidar_batch is None) and (radar_ray_bundle is None or radar_batch is None):
        ray_bundle, batch = img_ray_bundle, img_batch
    elif (lidar_ray_bundle is None or lidar_batch is None) and (img_ray_bundle is None or img_batch is None):
        ray_bundle, batch = radar_ray_bundle, radar_batch
    else:
        ray_bundle = img_ray_bundle.cat(lidar_ray_bundle, dim=0)
        ray_bundle = ray_bundle.cat(radar_ray_bundle, dim=0)
        overlapping_keys = set(img_batch.keys()).intersection(set(lidar_batch.keys())).intersection(
            set(radar_batch.keys())
        ) - {"is_lidar", "is_radar", "did_return"}
        assert not overlapping_keys, f"Overlapping keys in batch: {overlapping_keys}"
        batch = {
            **img_batch,
            **lidar_batch,
            **radar_batch,
            "is_lidar": torch.cat([img_batch["is_lidar"], lidar_batch["is_lidar"], radar_batch["is_lidar"]]),
            "is_radar": torch.cat([img_batch["is_radar"], lidar_batch["is_radar"], radar_batch["is_radar"]]),
            "did_return": torch.cat([img_batch["did_return"], lidar_batch["did_return"], radar_batch["did_return"]]),
        }
    return ray_bundle, batch
