
<p align="center">
    <!-- paper badges -->
    <a href="https://arxiv.org/abs/2504.00859">
        <img src='https://img.shields.io/badge/arXiv-Page-aff'>
    </a>
</p>

<div align="center">
<h4 style="font-size:1.5em;">
NeuRadar: Neural Radiance Fields for Automotive Radar Point Clouds
</h4>
</div>
<div align="center">

<div align="center">
<picture>
    <img alt="top figure" src="docs/imgs/neuradar_top_figure.png" width="120%">
</picture>
</div>

</div>

<h4>Code to be released.</h4>

# About
This is the official repository for _NeuRadar: Neural Radiance Fields for Automotive Radar Point Clouds_.

### Abstract
Radar is an important sensor for autonomous driving (AD) systems due to its robustness to adverse weather and different lighting conditions. Novel view synthesis using neural radiance fields (NeRFs) has recently received considerable attention in AD due to its potential to enable efficient testing and validation but remains unexplored for radar point clouds. In this paper, we present NeuRadar, a NeRF-based model that jointly generates radar point clouds, camera images, and lidar point clouds. We explore set-based object detection methods such as DETR, and propose an encoder-based solution grounded in the NeRF geometry for improved generalizability. We propose both a deterministic and a probabilistic point cloud representation to accurately model the radar behavior, with the latter being able to capture radar's stochastic behavior. We achieve realistic reconstruction results for two automotive datasets, establishing a baseline for NeRF-based radar point cloud simulation models. In addition, we release radar data for ZOD's Sequences and Drives to enable further research in this field.

# TODOs
- [ ] Release code

## Citation
```bibtex
@article{rafidashti2025neuradar,
  title        = {NeuRadar: Neural Radiance Fields for Automotive Radar Point Clouds},
  author       = {Rafidashti, Mahan and Lan, Ji and Fatemi, Maryam and Fu, Junsheng and Hammarstrand, Lars and Svensson, Lennart},
  journal      = {},
  year         = {2025}
}
