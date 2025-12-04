# Copyright 2023 The Nerfstudio Team. All rights reserved.
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
Lidar Dataset.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import torch

from nerfstudio.cameras.radars import Radars
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import Dataset


class RadarDataset(Dataset):
    """Dataset that returns lidar data.

    Args:
        dataparser_outputs: description of where and how to read data.
        downsample_factor: The downsample factor for the dataparser outputs (lidar only)
    """

    exclude_batch_keys_from_device: List[str] = []
    radars: Radars

    def __init__(self, dataparser_outputs: DataparserOutputs) -> None:
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.radars: Radars = deepcopy(self.metadata.pop("radars"))
        self.point_clouds = deepcopy(self.metadata.pop("radar_pcs"))
        self.has_masks = dataparser_outputs.mask_filenames is not None
        self.scene_box = deepcopy(dataparser_outputs.scene_box)

    def __len__(self):
        return len(self.radars)

    # pylint: disable=no-self-use
    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        return {}

    def get_data(self, radar_idx: int) -> Dict:
        """Returns the RadarDataset data as a dictionary.

        Args:
            radar_idx: The radar index in the dataset.
        """
        data = {"radar_idx": radar_idx}
        sensor_index = self.radars.metadata["sensor_idxs"].squeeze()[radar_idx] if len(self.radars.metadata["sensor_idxs"]) > 1 else torch.tensor([self.radars.metadata["sensor_idxs"]])
        data.update({"sensor_idx": sensor_index})
        point_cloud = self.point_clouds[radar_idx]
        point_cloud = torch.cat([point_cloud, torch.zeros(point_cloud.shape[0], 1)], dim=1)
        data["radar"] = point_cloud

        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def __getitem__(self, radar_idx: int) -> Dict:
        data = self.get_data(radar_idx)
        return data
