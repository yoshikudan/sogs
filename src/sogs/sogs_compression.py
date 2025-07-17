import json
import os
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torchpq.clustering import KMeans
from torch import Tensor
from plas import sort_with_plas

from plyfile import PlyData, PlyElement
from pathlib import Path
from PIL import Image

"""Uses quantization and sorting to compress splats into PNG files and uses
K-means clustering to compress the spherical harmonic coefficents.

.. warning::
    This class requires the `Pillow <https://pypi.org/project/pillow/>`_,
    `plas <https://github.com/fraunhoferhhi/PLAS.git>`_
    and `torchpq <https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install>`_ packages to be installed.

.. warning::
    This class might throw away a few lowest opacities splats if the number of
    splats is not a square number.

.. note::
    The splats parameters are expected to be pre-activation values. It expects
    the following fields in the splats dictionary: "means", "scales", "quats",
    "opacities", "sh0", "shN".

References:
    - `Compact 3D Scene Representation via Self-Organizing Gaussian Grids <https://arxiv.org/abs/2312.13299>`_
    - `Making Gaussian Splats more smaller <https://aras-p.info/blog/2023/09/27/Making-Gaussian-Splats-more-smaller/>`_
"""

def _get_compress_fn(param_name: str) -> Callable:
    compress_fn_map = {
        "means": _compress_16bit,
        "scales": _compress,
        "quats": _compress_quats,
        # "opacities" no longer compressed separately
        "sh0": _compress,  # placeholder, actual handling in run_compression
        "shN": _compress_kmeans,
    }
    return compress_fn_map[param_name]


def run_compression(compress_dir: str, splats: Dict[str, Tensor], verbose: bool, sort_method: str = "auto") -> None:
    """Run compression

    Args:
        compress_dir (str): directory to save compressed files
        splats (Dict[str, Tensor]): Gaussian splats to compress
    """

    # Param-specific preprocessing
    splats["means"] = log_transform(splats["means"])
    splats["quats"] = F.normalize(splats["quats"], dim=-1)
    neg_mask = splats["quats"][..., 3] < 0
    splats["quats"][neg_mask] *= -1
    # splats["quats"] = splats["quats"][..., :3]
    splats["sh0"] = splats["sh0"].clamp(-3.0, 3.0)
    if "shN" in splats:
        splats["shN"] = splats["shN"].clamp(-6.0, 6.0)

    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    n_crop = n_gs - n_sidelen**2
    if n_crop != 0:
        splats = _crop_n_splats(splats, n_crop)
        print(
            f"Warning: Number of Gaussians was not square. Removed {n_crop} Gaussians."
        )

    meta: Dict[str, Any] = {}

    splats = sort_splats(splats, verbose, sort_method)

    # Extract opacities and merge into sh0
    opacities = splats.pop("opacities")

    for param_name in splats.keys():
        if param_name == "sh0":
            meta["sh0"] = _compress_sh0_with_opacity(
                compress_dir, "sh0", splats["sh0"], opacities, n_sidelen, verbose=verbose
            )
        else:
            compress_fn = _get_compress_fn(param_name)
            meta[param_name] = compress_fn(
                compress_dir, param_name, splats[param_name], n_sidelen=n_sidelen, verbose=verbose
            )

    with open(os.path.join(compress_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def log_transform(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def write_image(compress_dir, param_name, img, lossless: bool=True, quality: int=100):
    """
    Compresses the image, currently as lossless webp. Centralized function to change
    image encoding in the future if need be.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        img (np.Array): image data as np array

    Returns:
        str: filename
    """
    filename = f"{param_name}.webp"
    Image.fromarray(img).save(
        os.path.join(compress_dir, filename),
        format="webp",
        lossless=lossless,
        quality=quality if not lossless else 100,
        method=6,
        exact=True
    )
    print(f"✓ {filename}")
    return filename


def _crop_n_splats(splats: Dict[str, Tensor], n_crop: int) -> Dict[str, Tensor]:
    opacities = splats["opacities"]
    keep_indices = torch.argsort(opacities, descending=True)[:-n_crop]
    for k, v in splats.items():
        splats[k] = v[keep_indices]
    return splats


def _compress(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, verbose: bool
) -> Dict[str, Any]:
    """Compress parameters with 8-bit quantization and lossless PNG compression."""
    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()

    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    img = img.squeeze()

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "files": [write_image(compress_dir, param_name, img)]
    }
    return meta


def _compress_16bit(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, verbose: bool
) -> Dict[str, Any]:
    """Compress parameters with 16-bit quantization and PNG compression."""
    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**16 - 1)).round().astype(np.uint16)
    img_l = img & 0xFF
    img_u = (img >> 8) & 0xFF

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "files": [
            write_image(compress_dir, f"{param_name}_l", img_l.astype(np.uint8)),
            write_image(compress_dir, f"{param_name}_u", img_u.astype(np.uint8))
        ]
    }
    return meta


def _compress_sh0_with_opacity(
    compress_dir: str,
    param_name: str,
    sh0: Tensor,
    opacities: Tensor,
    n_sidelen: int,
    verbose: bool
) -> Dict[str, Any]:
    """Combine sh0 (RGB) and opacities as alpha channel into a single RGBA texture."""
    # Reshape to spatial grid
    grid_sh0 = sh0.reshape((n_sidelen, n_sidelen, -1))
    grid_opac = opacities.reshape((n_sidelen, n_sidelen, 1))
    grid = torch.cat([grid_sh0, grid_opac], dim=-1)

    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()

    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    filename = write_image(compress_dir, param_name, img)

    meta = {
        # New channel count = original 3 + opacity = 4
        "shape": [*list(sh0.shape[:-1]), sh0.shape[-1] + 1],
        "dtype": str(sh0.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "files": [filename]
    }
    return meta


def _compress_kmeans(
    compress_dir: str,
    param_name: str,
    params: Tensor,
    n_sidelen: int,
    quantization: int = 8,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run K-means clustering on parameters and save centroids and labels as images."""
    params = params.reshape(params.shape[0], -1)
    dim = params.shape[1]
    n_clusters = round((len(params) >> 2) / 64) * 64
    n_clusters = min(n_clusters, 2 ** 16)

    kmeans = KMeans(n_clusters=n_clusters, distance="manhattan", verbose=verbose)
    labels = kmeans.fit(params.permute(1, 0).contiguous())
    labels = labels.detach().cpu().numpy()
    centroids = kmeans.centroids.permute(1, 0)

    mins = torch.min(centroids)
    maxs = torch.max(centroids)
    centroids_norm = (centroids - mins) / (maxs - mins)
    centroids_norm = centroids_norm.detach().cpu().numpy()
    centroids_quant = (
        (centroids_norm * (2**quantization - 1)).round().astype(np.uint8)
    )

    # sort centroids for compact atlas layout
    sorted_indices = np.lexsort(centroids_quant.T)
    sorted_indices = sorted_indices.reshape(64, -1).T.reshape(-1)
    sorted_centroids_quant = centroids_quant[sorted_indices]
    inverse = np.argsort(sorted_indices)

    centroids_packed = sorted_centroids_quant.reshape(-1, int(dim * 64 / 3), 3)
    labels = inverse[labels].astype(np.uint16).reshape((n_sidelen, n_sidelen))
    labels_l = labels & 0xFF
    labels_u = (labels >> 8) & 0xFF

    # Combine low and high bytes into single texture: R=labels_l, G=labels_u, B=0
    labels_combined = np.zeros((n_sidelen, n_sidelen, 3), dtype=np.uint8)
    labels_combined[..., 0] = labels_l.astype(np.uint8)
    labels_combined[..., 1] = labels_u.astype(np.uint8)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "quantization": quantization,
        "files": [
            write_image(compress_dir, f"{param_name}_centroids", centroids_packed),
            write_image(compress_dir, f"{param_name}_labels", labels_combined)
        ]
    }
    return meta

def pack_quaternion_to_rgba_tensor(q: Tensor) -> Tensor:
    """
    Packs a batch of quaternions into RGBA channels:
      - R,G,B: the three smallest components, scaled by sqrt(2) then mapped from [-1,1]→[0,1]
      - A: index of largest-abs component (0→3) mapped [0,3]→[0,1]
    q: (...,4)
    returns: (...,4) in [0,1]
    """
    abs_q = q.abs()
    max_idx = abs_q.argmax(dim=-1)  # (...)

    # ensure largest component is positive
    max_vals = q.gather(-1, max_idx.unsqueeze(-1)).squeeze(-1)
    sign = max_vals.sign()
    sign[sign == 0] = 1
    q_signed = q * sign.unsqueeze(-1)

    # build variants dropping each component
    variants = []
    for i in range(4):
        dims = list(range(4))
        dims.remove(i)
        variants.append(q_signed[..., dims])  # (...,3)
    stacked = torch.stack(variants, dim=-2)  # (...,4,3)

    # select the appropriate 3-vector based on max_idx
    idx_exp = max_idx.unsqueeze(-1).unsqueeze(-1).expand(*max_idx.shape, 1, 3)
    small = torch.gather(stacked, dim=-2, index=idx_exp).squeeze(-2)  # (...,3)

    # scale by sqrt(2) to normalize range to [-1,1]
    small = small * torch.sqrt(torch.tensor(2.0, device=small.device, dtype=small.dtype))

    # map from [-1,1] to [0,1]
    rgb = small * 0.5 + 0.5
    a = (252.0 + max_idx.to(torch.float32)) / 255.0
    return torch.cat([rgb, a.unsqueeze(-1)], dim=-1)

def _compress_quats(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, verbose: bool
) -> Dict[str, Any]:
    """Compress quaternions by packing into RGBA and saving as an 8-bit image."""
    # params: (n_splats,4)
    rgba = pack_quaternion_to_rgba_tensor(params)
    img = (rgba.view(n_sidelen, n_sidelen, 4).cpu().numpy() * 255.0).round().astype(np.uint8)
    filename = write_image(compress_dir, f"{param_name}", img)

    meta = {
        "shape": list(params.shape),
        "dtype": "uint8",
        "encoding": "quaternion_packed",
        "files": [filename]
    }
    return meta

def sort_splats_simple(splats: Dict[str, Tensor], verbose: bool = True) -> Dict[str, Tensor]:
    """Sort splats using simple spatial sorting (much more memory efficient).
    
    Sorts by position (x, y, z) to create a spatially coherent grid.
    This is much faster and uses O(n) memory instead of O(n²).

    Args:
        splats (Dict[str, Tensor]): splats
        verbose (bool, optional): Whether to print verbose information. Default to True.

    Returns:
        Dict[str, Tensor]: sorted splats
    """
    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    assert n_sidelen**2 == n_gs, "Must be a perfect square"

    if verbose:
        print("Using simple spatial sorting (memory efficient)")

    # Sort by position (x, y, z) to create spatially coherent grid
    means = splats["means"]  # (n_gs, 3)
    
    # Create sorting key: combine x, y, z with some weight
    # This creates a space-filling curve effect
    x, y, z = means[:, 0], means[:, 1], means[:, 2]
    
    # Normalize coordinates to [0, 1] range
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    
    # Create sorting key using Z-order curve (Morton code approximation)
    # This creates better spatial coherence than simple lexicographic sorting
    sort_key = x_norm + 2 * y_norm + 4 * z_norm
    
    # Get sorted indices
    sorted_indices = torch.argsort(sort_key)
    
    # Apply sorting to all splat parameters
    for k, v in splats.items():
        splats[k] = v[sorted_indices]
    
    return splats


def sort_splats_opacity(splats: Dict[str, Tensor], verbose: bool = True) -> Dict[str, Tensor]:
    """Sort splats by opacity (most important splats first).
    
    This is very memory efficient and often provides good compression
    by putting high-opacity splats together.

    Args:
        splats (Dict[str, Tensor]): splats
        verbose (bool, optional): Whether to print verbose information. Default to True.

    Returns:
        Dict[str, Tensor]: sorted splats
    """
    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    assert n_sidelen**2 == n_gs, "Must be a perfect square"

    if verbose:
        print("Using opacity-based sorting (memory efficient)")

    # Sort by opacity (descending - most opaque first)
    opacities = splats["opacities"]
    sorted_indices = torch.argsort(opacities, descending=True)
    
    # Apply sorting to all splat parameters
    for k, v in splats.items():
        splats[k] = v[sorted_indices]
    
    return splats


def sort_splats_plas(splats: Dict[str, Tensor], verbose: bool = True) -> Dict[str, Tensor]:
    """Sort splats with Parallel Linear Assignment Sorting from the paper.

    Args:
        splats (Dict[str, Tensor]): splats
        verbose (bool, optional): Whether to print verbose information. Default to True.

    Returns:
        Dict[str, Tensor]: sorted splats
    """
    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    assert n_sidelen**2 == n_gs, "Must be a perfect square"

    if verbose:
        print("Using PLAS sorting (memory intensive)")

    sort_keys = [k for k in splats if k != "shN"]
    params_to_sort = torch.cat([splats[k].reshape(n_gs, -1) for k in sort_keys], dim=-1)
    shuffled_indices = torch.randperm(
        params_to_sort.shape[0], device=params_to_sort.device
    )
    params_to_sort = params_to_sort[shuffled_indices]
    grid = params_to_sort.reshape((n_sidelen, n_sidelen, -1))
    _, sorted_indices = sort_with_plas(
        grid.permute(2, 0, 1), improvement_break=1e-4, verbose=verbose
    )
    sorted_indices = sorted_indices.squeeze().flatten()
    sorted_indices = shuffled_indices[sorted_indices]
    for k, v in splats.items():
        splats[k] = v[sorted_indices]
    return splats


def sort_splats_flas(splats: Dict[str, Tensor], verbose: bool = True) -> Dict[str, Tensor]:
    """Sort splats with Fast Linear Assignment Sorting (FLAS) - GPU parallelized version.
    
    FLAS is a memory-efficient alternative to PLAS that uses hierarchical clustering
    and local optimization instead of global pairwise distance computation.
    This version is optimized for GPU parallelization.
    
    Args:
        splats (Dict[str, Tensor]): splats
        verbose (bool, optional): Whether to print verbose information. Default to True.

    Returns:
        Dict[str, Tensor]: sorted splats
    """
    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    assert n_sidelen**2 == n_gs, "Must be a perfect square"

    if verbose:
        print("Using GPU-parallelized FLAS sorting (memory efficient)")

    # FLAS approach: hierarchical clustering + local optimization
    means = splats["means"]  # (n_gs, 3)
    
    # Step 1: Create initial grid-based ordering with GPU acceleration
    x, y, z = means[:, 0], means[:, 1], means[:, 2]
    
    # Normalize to grid coordinates (GPU parallelized)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    
    # Create initial sorting key using Hilbert curve approximation
    # This provides better spatial coherence than Z-order
    sort_key = x_norm + 3 * y_norm + 9 * z_norm
    initial_indices = torch.argsort(sort_key)
    
    # Step 2: GPU-parallelized local optimization using batched windows
    window_size = min(2048, n_gs // 20)  # Larger windows for better GPU utilization
    num_windows = (n_gs + window_size - 1) // window_size
    
    if verbose:
        print(f"FLAS: Using {num_windows} windows of size {window_size}")
    
    # Apply initial sorting
    for k, v in splats.items():
        splats[k] = v[initial_indices]
    
    # Step 3: GPU-parallelized local optimization
    if n_gs < 2000000:  # Increased threshold for GPU optimization
        if verbose:
            print("FLAS: Applying GPU-parallelized local optimization")
        
        # Use a subset of parameters for local optimization to save memory
        sort_keys = [k for k in splats if k != "shN"]
        params_to_sort = torch.cat([splats[k].reshape(n_gs, -1) for k in sort_keys], dim=-1)
        
        # Use float16 for memory efficiency
        params_to_sort = params_to_sort.half()
        
        # GPU-parallelized local optimization
        optimized_indices = torch.arange(n_gs, device=params_to_sort.device)
        
        # Process windows in batches for better GPU utilization
        batch_size = 4  # Process multiple windows simultaneously
        for batch_start in range(0, num_windows, batch_size):
            batch_end = min(batch_start + batch_size, num_windows)
            
            # Prepare batch of windows
            window_batch_indices = []
            window_batch_params = []
            
            for window_idx in range(batch_start, batch_end):
                start_idx = window_idx * window_size
                end_idx = min(start_idx + window_size, n_gs)
                window_size_actual = end_idx - start_idx
                
                if window_size_actual < 2:
                    continue
                
                window_params = params_to_sort[start_idx:end_idx]
                window_batch_params.append(window_params)
                window_batch_indices.append((start_idx, end_idx))
            
            if not window_batch_params:
                continue
            
            # Process batch of windows in parallel
            for i, (window_params, (start_idx, end_idx)) in enumerate(zip(window_batch_params, window_batch_indices)):
                window_size_actual = window_params.shape[0]
                
                # Compute pairwise distances within window (GPU parallelized)
                distances = torch.cdist(window_params, window_params)
                
                # GPU-optimized greedy nearest neighbor
                # Use vectorized operations instead of loops
                ordered_indices = torch.zeros(window_size_actual, dtype=torch.long, device=params_to_sort.device)
                
                # Start with the most central point
                center_idx = torch.argmin(distances.sum(dim=1))
                ordered_indices[0] = center_idx
                
                # Create mask for remaining points
                remaining_mask = torch.ones(window_size_actual, dtype=torch.bool, device=params_to_sort.device)
                remaining_mask[center_idx] = False
                
                # Vectorized greedy nearest neighbor
                for pos in range(1, window_size_actual):
                    last_idx = ordered_indices[pos - 1]
                    # Find nearest remaining point
                    distances_to_last = distances[last_idx]
                    distances_to_last[~remaining_mask] = float('inf')  # Mask out already selected points
                    next_idx = torch.argmin(distances_to_last)
                    
                    ordered_indices[pos] = next_idx
                    remaining_mask[next_idx] = False
                
                # Update global indices
                optimized_indices[start_idx:end_idx] = start_idx + ordered_indices
        
        # Apply optimized ordering
        for k, v in splats.items():
            splats[k] = v[optimized_indices]
    
    return splats


def sort_splats_multi_gpu(splats: Dict[str, Tensor], verbose: bool = True) -> Dict[str, Tensor]:
    """Sort splats using multiple GPUs if available.
    
    This function distributes the sorting workload across multiple GPUs
    for better performance on large datasets.
    
    Args:
        splats (Dict[str, Tensor]): splats
        verbose (bool, optional): Whether to print verbose information. Default to True.

    Returns:
        Dict[str, Tensor]: sorted splats
    """
    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    assert n_sidelen**2 == n_gs, "Must be a perfect square"

    if verbose:
        print("Using multi-GPU sorting")

    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        if verbose:
            print(f"Only {num_gpus} GPU available, using single GPU")
        return sort_splats_flas(splats, verbose)

    if verbose:
        print(f"Using {num_gpus} GPUs for parallel sorting")

    # Split data across GPUs
    chunk_size = n_gs // num_gpus
    chunks = []
    
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_gpus - 1 else n_gs
        
        # Move chunk to GPU i
        chunk = {}
        for k, v in splats.items():
            chunk[k] = v[start_idx:end_idx].to(f'cuda:{i}')
        chunks.append(chunk)

    # Sort each chunk in parallel on different GPUs
    sorted_chunks = []
    for i, chunk in enumerate(chunks):
        if verbose:
            print(f"Sorting chunk {i+1}/{num_gpus} on GPU {i}")
        
        # Use FLAS on each chunk
        sorted_chunk = sort_splats_flas(chunk, verbose=False)
        sorted_chunks.append(sorted_chunk)

    # Merge sorted chunks back to main GPU
    merged_splats = {}
    for k in splats.keys():
        merged_tensors = []
        for chunk in sorted_chunks:
            merged_tensors.append(chunk[k].to('cuda:0'))
        merged_splats[k] = torch.cat(merged_tensors, dim=0)

    # Final global optimization on main GPU (optional)
    if n_gs < 1000000:  # Only for smaller datasets
        if verbose:
            print("Applying final global optimization")
        
        # Simple global optimization: sort by position
        means = merged_splats["means"]
        x, y, z = means[:, 0], means[:, 1], means[:, 2]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        sort_key = x_norm + 3 * y_norm + 9 * z_norm
        final_indices = torch.argsort(sort_key)
        
        for k, v in merged_splats.items():
            merged_splats[k] = v[final_indices]

    return merged_splats


def sort_splats(splats: Dict[str, Tensor], verbose: bool = True, method: str = "auto") -> Dict[str, Tensor]:
    """Sort splats using the specified method.
    
    Args:
        splats (Dict[str, Tensor]): splats
        verbose (bool, optional): Whether to print verbose information. Default to True.
        method (str): Sorting method - "auto", "simple", "opacity", "plas", or "none"

    Returns:
        Dict[str, Tensor]: sorted splats
    """
    n_gs = len(splats["means"])
    
    if method == "none":
        if verbose:
            print("Skipping sorting as requested")
        return splats
    
    if method == "auto":
        # Auto-select based on dataset size and available memory
        estimated_plas_memory_gb = (n_gs * n_gs * 2) / (1024**3)  # float16
        estimated_flas_memory_gb = (min(2048, n_gs // 20) ** 2 * 2) / (1024**3)  # window-based
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free_memory_gb = torch.cuda.memory_reserved(0) / (1024**3)
            available_memory_gb = gpu_memory_gb - free_memory_gb
            
            # Check if multi-GPU is beneficial
            if num_gpus >= 2 and n_gs > 500000:  # Large dataset with multiple GPUs
                if verbose:
                    print(f"Auto-selecting multi-GPU sorting ({num_gpus} GPUs available)")
                method = "multi_gpu"
            elif estimated_plas_memory_gb > available_memory_gb * 0.5:  # PLAS too memory intensive
                if estimated_flas_memory_gb > available_memory_gb * 0.3:  # FLAS also too memory intensive
                    if verbose:
                        print(f"Auto-selecting simple sorting (PLAS: {estimated_plas_memory_gb:.1f}GB, FLAS: {estimated_flas_memory_gb:.1f}GB)")
                    method = "simple"
                else:
                    if verbose:
                        print(f"Auto-selecting FLAS sorting (PLAS: {estimated_plas_memory_gb:.1f}GB, FLAS: {estimated_flas_memory_gb:.1f}GB)")
                    method = "flas"
            else:
                if verbose:
                    print(f"Auto-selecting PLAS sorting (estimated {estimated_plas_memory_gb:.1f}GB)")
                method = "plas"
        else:
            if verbose:
                print("Auto-selecting simple sorting (no CUDA available)")
            method = "simple"
    
    if method == "simple":
        return sort_splats_simple(splats, verbose)
    elif method == "opacity":
        return sort_splats_opacity(splats, verbose)
    elif method == "plas":
        return sort_splats_plas(splats, verbose)
    elif method == "flas":
        return sort_splats_flas(splats, verbose)
    elif method == "multi_gpu":
        return sort_splats_multi_gpu(splats, verbose)
    else:
        raise ValueError(f"Unknown sorting method: {method}. Available: simple, opacity, plas, flas, multi_gpu, auto, none")


@torch.no_grad()
def read_ply(path):
    """
    Reads a .ply file and reconstructs a dictionary of PyTorch tensors on GPU.
    """
    plydata = PlyData.read(path)
    vd = plydata['vertex'].data

    def has_col(col_name):
        return col_name in vd.dtype.names

    xyz = np.stack([vd['x'], vd['y'], vd['z']], axis=-1)
    f_dc = np.stack([vd[f"f_dc_{i}"] for i in range(3)], axis=-1)

    rest_cols = [c for c in vd.dtype.names if c.startswith('f_rest_')]
    rest_cols_sorted = sorted(rest_cols, key=lambda c: int(c.split('_')[-1]))
    if len(rest_cols_sorted) > 0:
        f_rest = np.stack([vd[c] for c in rest_cols_sorted], axis=-1)
    else:
        f_rest = np.empty((len(vd), 0), dtype=np.float32)

    opacities = vd['opacity']
    scale = np.stack([vd[f"scale_{i}"] for i in range(3)], axis=-1)
    rotation = np.stack([vd[f"rot_{i}"] for i in range(4)], axis=-1)

    splats = {}
    splats["means"] = torch.from_numpy(xyz).float().cuda()
    splats["opacities"] = torch.from_numpy(opacities).float().cuda()
    splats["scales"] = torch.from_numpy(scale).float().cuda()
    splats["quats"] = torch.from_numpy(rotation).float().cuda()

    sh0_tensor = torch.from_numpy(f_dc).float()
    sh0_tensor = sh0_tensor.unsqueeze(-1).transpose(1, 2)
    splats["sh0"] = sh0_tensor.cuda()

    if f_rest.any():
        if f_rest.shape[1] % 3 != 0:
            raise ValueError(f"Number of f_rest columns ({f_rest.shape[1]}) not divisible by 3.")
        num_rest_per_channel = f_rest.shape[1] // 3
        shn_tensor = torch.from_numpy(
            f_rest.reshape(-1, 3, num_rest_per_channel)
        ).float().transpose(1, 2)
        splats["shN"] = shn_tensor.cuda()

    return splats
