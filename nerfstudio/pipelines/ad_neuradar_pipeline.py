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

from dataclasses import dataclass, field
from typing import Type

import torch
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from nerfstudio.data.datamanagers.ad_neuradar_datamanager import ADNeuRadarDataManager, ADNeuRadarDataManagerConfig
from nerfstudio.models.ad_model import ADModel
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.pipelines.ad_pipeline import ADPipeline, ADPipelineConfig
from nerfstudio.utils import profiler


@dataclass
class ADNeuRadarPipelineConfig(ADPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: ADNeuRadarPipeline)
    """target class to instantiate"""
    datamanager: ADNeuRadarDataManagerConfig = field(default_factory=ADNeuRadarDataManagerConfig)
    """specifies the datamanager config"""

class ADNeuRadarPipeline(ADPipeline):
    """Pipeline for training AD models."""

    def __init__(self, config: ADNeuRadarPipelineConfig, **kwargs):
        pixel_sampler = config.datamanager.pixel_sampler
        pixel_sampler.patch_size = config.ray_patch_size[0]
        pixel_sampler.patch_scale = config.model.rgb_upsample_factor
        VanillaPipeline.__init__(self, config, **kwargs)

        # Fix type hints
        self.datamanager: ADNeuRadarDataManager = self.datamanager
        self.model: ADModel = self.model
        self.config: ADNeuRadarPipelineConfig = self.config

        # Disable ray drop classification if we do not add missing points
        if not self.datamanager.dataparser.config.add_missing_points:
            self.model.disable_ray_drop()

        self.fid = None

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        self.model.set_active_levels()
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle, batch, patch_size=self.config.ray_patch_size)
        metrics_dict, radar_dict = self.model.get_metrics_dict(model_outputs, batch)

        if (actors := self.model.dynamic_actors).config.optimize_trajectories:
            pos_norm = (actors.actor_positions - actors.initial_positions).norm(dim=-1)
            metrics_dict["traj_opt_translation"] = pos_norm[pos_norm > 0].mean().nan_to_num()
            metrics_dict["traj_opt_rotation"] = (
                (actors.actor_rotations_6d - actors.initial_rotations_6d)[pos_norm > 0].norm(dim=-1).mean().nan_to_num()
            )

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict, radar_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle, batch, patch_size=self.config.ray_patch_size)
        metrics_dict, radar_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict, radar_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        # Image eval
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera(camera)
        metrics_dict, images_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()

        # Lidar eval
        lidar, batch = self.datamanager.next_eval_lidar(step)
        outputs, batch = self.model.get_outputs_for_lidar(lidar, batch=batch)
        lidar_metrics_dict, _, _ = self.model.get_image_metrics_and_images(outputs, batch)
        assert not set(lidar_metrics_dict.keys()).intersection(metrics_dict.keys())
        metrics_dict.update(lidar_metrics_dict)

        # radar eval
        radar_idx, radar_ray_bundle, radar_batch = self.datamanager.next_eval_radar(step)
        radar_outputs = self.model.get_outputs_for_camera_ray_bundle(radar_ray_bundle)
        r2w = self.datamanager.eval_radar_dataset.radars.radar_to_worlds[radar_idx]
        radar_metrics_dict, _, radar_dict = self.model.get_image_metrics_and_images(radar_outputs, radar_batch, self.datamanager.eval_radar_dataset.radars.times, radar_idx, r2w)
        assert "radar_idx" not in radar_metrics_dict
        radar_metrics_dict["radar_idx"] = radar_batch['radar_idx']
        assert not set(radar_metrics_dict.keys()).intersection(metrics_dict.keys())
        metrics_dict.update(radar_metrics_dict)

        self.train()
        return metrics_dict, images_dict, radar_dict

    @profiler.time_function
    def get_average_eval_radar_metrics(self):
        self.eval()
        print("start eval all radar!")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            num_radar = len(self.datamanager.fixed_indices_eval_radar_dataloader)
            device = self.datamanager.fixed_indices_eval_radar_dataloader.device
            sampling_rounds = 1 if self.model.config.loss.radar_loss_type == "euclidean" else 10
            idx = 0
            metrics_dict_arr = {"chamfer_distance": torch.zeros((num_radar, sampling_rounds)).to(device),
                                "emd_distance": torch.zeros((num_radar, sampling_rounds)).to(device),}

            task = progress.add_task("[green]Evaluating all eval radar point clouds...", total=num_radar)
            for _, batch in self.datamanager.fixed_indices_eval_radar_dataloader:

                radar_idx = batch['radar_idx']
                num_points = batch['radar'].shape[0]
                ray_indices = torch.cat([torch.full((num_points, 1), radar_idx, dtype=torch.int64, device=self.device), torch.arange(num_points, device=self.device).view(-1, 1)], dim=-1)
                batch['indices'] = ray_indices
                scan_index = torch.tensor([radar_idx])
                ray_bundle = self.datamanager.eval_radar_dataset.radars.generate_rays(scan_indices=scan_index).to(self.device)

                ray_bundle.metadata["is_radar"] = torch.ones((*ray_bundle.shape, 1), device=device, dtype=torch.bool)
                outputs = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)
                batch["distance"] = ray_bundle.metadata["directions_norm"]

                metrics_dict = self.model.get_radar_metrics(outputs, batch, sampling_rounds) # type: ignore
                for key in metrics_dict_arr.keys():
                    metrics_dict_arr[key][idx] = metrics_dict[key]
                idx += 1
                progress.advance(task)

        self.train()
        avg_eval_radar_metrics = {}
        for key in metrics_dict_arr.keys():
            avg_eval_radar_metrics[key + '_mean'] = metrics_dict_arr[key].mean()
            avg_eval_radar_metrics[key + '_median'] = metrics_dict_arr[key].median()
            avg_eval_radar_metrics[key + '_std'] = metrics_dict_arr[key].mean(dim=1).std()
        return avg_eval_radar_metrics
