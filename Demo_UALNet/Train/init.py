import os
import shutil
from os.path import join
from typing import Iterable, Tuple

import torch
from torch.utils.data import DataLoader

from dataloader import LoadDataPair as InpDataloader


def init_optimizer(
    parameters,
    lr: float,
    optimizer_type: str = "Adam",
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Initialize optimizer and learning rate scheduler.

    Args:
        parameters: Model parameters.
        lr (float): Learning rate.
        optimizer_type (str): Optimizer type. Supported:
            - "Adam"
            - "SGD"
            - "RMSprop"

    Returns:
        optimizer: Initialized optimizer.
        scheduler: Cosine annealing learning rate scheduler.
    """
    if lr <= 0:
        raise ValueError(f"Learning rate must be positive, but got {lr}.")

    optimizer_type = optimizer_type.strip()

    if optimizer_type == "RMSprop":
        optimizer = torch.optim.RMSprop(parameters, lr=lr)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(parameters, lr=lr)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(parameters, lr=lr)
    else:
        raise ValueError(
            f"Unsupported optimizer type: '{optimizer_type}'. "
            f"Supported types are ['Adam', 'SGD', 'RMSprop']."
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=500,
        eta_min=5e-5,
        last_epoch=-1,
    )

    return optimizer, scheduler


def init_data(
    train_data_path: str,
    test_data_path: str,
    valid_data_path: str,
    batch_size: int,
    data_size,
    num_workers: int = 10,
    pin_memory: bool = False,
):
    """
    Initialize train / validation / test dataloaders.

    Args:
        train_data_path (str): Path to training data.
        test_data_path (str): Path to testing data.
        valid_data_path (str): Path to validation data.
        batch_size (int): Batch size for training loader.
        data_size: Input crop size or dataset size argument passed to dataset.
        num_workers (int): Number of workers for DataLoader.
        pin_memory (bool): Whether to enable pin_memory.

    Returns:
        train_loader, valid_loader, test_loader
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, but got {batch_size}.")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, but got {num_workers}.")

    for path_name, path_value in {
        "train_data_path": train_data_path,
        "test_data_path": test_data_path,
        "valid_data_path": valid_data_path,
    }.items():
        if not isinstance(path_value, str):
            raise TypeError(f"{path_name} must be a string, but got {type(path_value)}.")
        if not os.path.exists(path_value):
            raise FileNotFoundError(f"{path_name} does not exist: {path_value}")

    train_set = InpDataloader(train_data_path, data_size)
    test_set = InpDataloader(test_data_path, data_size)
    valid_set = InpDataloader(valid_data_path, data_size)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader


def init_save(
    files_to_save: Iterable[str],
    save_path: str,
    save_results: str,
) -> None:
    """
    Create save directories and back up specified code files.

    Args:
        files_to_save (Iterable[str]): Filenames to copy.
        save_path (str): Directory for experiment outputs.
        save_results (str): Directory for saving result images / outputs.
    """
    if not isinstance(save_path, str):
        raise TypeError(f"save_path must be a string, but got {type(save_path)}.")
    if not isinstance(save_results, str):
        raise TypeError(f"save_results must be a string, but got {type(save_results)}.")

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_results, exist_ok=True)

    copy_files(
        files=files_to_save,
        base_dir="./",
        dst_dir=join(save_path, "code"),
    )


def copy_files(
    files: Iterable[str],
    base_dir: str,
    dst_dir: str,
) -> None:
    """
    Copy files from base_dir to dst_dir.

    Args:
        files (Iterable[str]): List of filenames to copy.
        base_dir (str): Source directory.
        dst_dir (str): Destination directory.
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    os.makedirs(dst_dir, exist_ok=True)

    for filename in files:
        src_path = join(base_dir, filename)
        dst_path = join(dst_dir, filename)

        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"Source file does not exist: {src_path}")

        shutil.copyfile(src_path, dst_path)