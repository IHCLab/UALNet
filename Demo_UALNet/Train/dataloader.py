import os
from os.path import join
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

class LoadDataPair(Dataset):
    """
    PyTorch dataset for loading hyperspectral / multispectral .mat files.

    Each .mat file is expected to contain:
        - 'Sentinel_multire' : real Sentinel-2 data
        - 'Sentinel_unified' : resolution-unified Sentinel-2 data simulated using real AVIRIS hyperspectral data
        - 'I256_c'           : real AVIRIS hyperspectral data

    Args:
        root_dir (str): Directory containing .mat files.
        crop_size (Optional[Tuple[int, int]]): Spatial crop size (H, W).
            If None, no cropping is applied.
        use_crop (bool): Whether to apply random cropping.
        normalize (bool): Whether to apply per-sample normalization.
        return_filename (bool): Whether to return the source filename.

    Returns:
        dict with keys:
            - 'Sentinel_2'
            - 'sen_simu'
            - 'AVIRIS'
            - 'fname' (optional)
    """

    SENTINEL_KEY = "Sentinel_multire"
    AVIRIS_KEY = "I256_c"
    SEN_SIMU_KEY = "Sentinel_unified"

    def __init__(
        self,
        root_dir: str,
        crop_size: Optional[Tuple[int, int]] = None,
        use_crop: bool = False,
        normalize: bool = False,
        return_filename: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(root_dir, str):
            raise TypeError(f"root_dir must be a string, but got {type(root_dir)}.")

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Directory does not exist: {root_dir}")

        self.root_dir = root_dir
        self.crop_size = crop_size
        self.use_crop = use_crop
        self.normalize = normalize
        self.return_filename = return_filename

        if self.use_crop:
            if crop_size is None:
                raise ValueError("crop_size must be provided when use_crop=True.")
            if (
                not isinstance(crop_size, (tuple, list))
                or len(crop_size) != 2
                or crop_size[0] <= 0
                or crop_size[1] <= 0
            ):
                raise ValueError(
                    f"crop_size must be a tuple/list of two positive integers, got {crop_size}."
                )

        self.filenames = sorted(
            [
                join(root_dir, fname)
                for fname in os.listdir(root_dir)
                if fname.endswith(".mat")
            ]
        )

        if len(self.filenames) == 0:
            raise RuntimeError(f"No .mat files found in directory: {root_dir}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not isinstance(idx, int):
            raise TypeError(f"Index must be int, but got {type(idx)}.")

        if idx < 0 or idx >= len(self.filenames):
            raise IndexError(
                f"Index {idx} is out of range for dataset of length {len(self.filenames)}."
            )

        file_path = self.filenames[idx]

        sentinel_2, aviris, sen_simu = self._load_mat(file_path)

        sentinel_2 = self._to_tensor(sentinel_2, name="Sentinel_2")
        aviris = self._to_tensor(aviris, name="AVIRIS")
        sen_simu = self._to_tensor(sen_simu, name="sen_simu")

        self._validate_shapes(sentinel_2, aviris, sen_simu, file_path)

        if self.use_crop:
            sentinel_2, aviris, sen_simu = self._random_crop_triplet(
                sentinel_2, aviris, sen_simu
            )

        if self.normalize:
            sentinel_2 = self._normalize(sentinel_2)
            aviris = self._normalize(aviris)
            sen_simu = self._normalize(sen_simu)

        sample = {
            "Sentinel_2": sentinel_2,
            "sen_simu": sen_simu,
            "AVIRIS": aviris,
        }

        if self.return_filename:
            sample["fname"] = file_path

        return sample

    def _load_mat(self, file_path: str):
        """
        Load a .mat file and extract required arrays.
        """
        try:
            mat_data = loadmat(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read .mat file: {file_path}\n{e}") from e

        required_keys = [
            self.SENTINEL_KEY,
            self.AVIRIS_KEY,
            self.SEN_SIMU_KEY,
        ]

        for key in required_keys:
            if key not in mat_data:
                raise KeyError(
                    f"Missing key '{key}' in file: {file_path}. "
                    f"Available keys: {list(mat_data.keys())}"
                )

        sentinel_2 = mat_data[self.SENTINEL_KEY]
        aviris = mat_data[self.AVIRIS_KEY]
        sen_simu = mat_data[self.SEN_SIMU_KEY]

        return sentinel_2, aviris, sen_simu

    def _to_tensor(self, array: np.ndarray, name: str) -> torch.Tensor:
        """
        Convert HWC numpy array to CHW float tensor.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray, but got {type(array)}.")

        if array.ndim != 3:
            raise ValueError(
                f"{name} must have shape (H, W, C), but got shape {array.shape}."
            )

        if np.isnan(array).any():
            raise ValueError(f"{name} contains NaN values.")

        if np.isinf(array).any():
            raise ValueError(f"{name} contains Inf values.")

        tensor = torch.from_numpy(array.astype(np.float32)).permute(2, 0, 1).contiguous()
        return tensor

    def _validate_shapes(
        self,
        sentinel_2: torch.Tensor,
        aviris: torch.Tensor,
        sen_simu: torch.Tensor,
        file_path: str,
    ) -> None:
        """
        Basic consistency checks for tensor shapes.
        """
        if sentinel_2.ndim != 3 or aviris.ndim != 3 or sen_simu.ndim != 3:
            raise ValueError(
                f"All tensors must be 3D (C, H, W). Got "
                f"Sentinel_2={tuple(sentinel_2.shape)}, "
                f"AVIRIS={tuple(aviris.shape)}, "
                f"sen_simu={tuple(sen_simu.shape)} "
                f"in file: {file_path}"
            )

        h_s2, w_s2 = sentinel_2.shape[1:]
        h_sim, w_sim = sen_simu.shape[1:]

        if h_sim != 2 * h_s2 or w_sim != 2 * w_s2:
            raise ValueError(
                f"Resolution mismatch in file: {file_path}. "
                f"Expected sen_simu spatial size to be 2x Sentinel_2.\n"
                f"Got Sentinel_2 = {tuple(sentinel_2.shape)}, "
                f"sen_simu = {tuple(sen_simu.shape)}"
            )
        
        # AVIRIS may have different channels, but if spatial size should match your setting,
        # this check is useful.
        if aviris.shape[1:] != sen_simu.shape[1:]:
            raise ValueError(
                f"Spatial size mismatch between AVIRIS and sen_simu in file: {file_path}. "
                f"Got AVIRIS={tuple(aviris.shape)}, "
                f"sen_simu={tuple(sen_simu.shape)}"
            )

    def _normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor using:
            x <- (x - min(x)) / ||x - min(x)||_2
        """
        data = data - torch.min(data)
        norm = torch.sqrt(torch.sum(data ** 2)).clamp_min(1e-12)
        data = data / norm
        return data

    def _random_crop_triplet(
        self,
        sentinel_2: torch.Tensor,
        aviris: torch.Tensor,
        sen_simu: torch.Tensor,
    ):
        """
        Apply the same random crop to all tensors.
        """
        _, h, w = sentinel_2.shape
        crop_h, crop_w = self.crop_size

        if crop_h > h or crop_w > w:
            raise ValueError(
                f"Crop size {self.crop_size} is larger than input size {(h, w)}."
            )

        if crop_h == h and crop_w == w:
            return sentinel_2, aviris, sen_simu

        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)

        sentinel_2 = sentinel_2[:, top : top + crop_h, left : left + crop_w]
        aviris = aviris[:, top : top + crop_h, left : left + crop_w]
        sen_simu = sen_simu[:, top : top + crop_h, left : left + crop_w]

        return sentinel_2, aviris, sen_simu


if __name__ == "__main__":
    dataset = LoadDataPair(root_dir="your_data_path", crop_size=(64, 64), use_crop=True)
    sample = dataset[0]
    print(sample["Sentinel_2"].shape)
    print(sample["sen_simu"].shape)
    print(sample["AVIRIS"].shape)
    print(sample["fname"])