# SOGS

Python package to compress Gaussian Splats with [Self-Organizing Gaussians](https://github.com/fraunhoferhhi/Self-Organizing-Gaussians)

Code forked from gsplat's [png_compression](https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/compression/png_compression.py) module and produces a compressed bundle suitable for rendering with PlayCanvas' SuperSplat.

## Usage

`pip install sogs`

`sogs-compress --input-ply your_ply_file.ply --output-dir directory_to_store_images_and_metadata`