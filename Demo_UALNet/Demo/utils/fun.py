import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def _prepare_rgb_cube(image: np.ndarray, band_indices: list[int], scale: float) -> np.ndarray:

    rgb_image = image[:, :, band_indices].astype(np.float32) * scale
    rgb_image = np.clip(rgb_image, 0.0, 1.0)
    return rgb_image


def plot_sentinel12_comparison(
    model: np.ndarray,
    sentinel_2: np.ndarray,
    save_path: str | Path,
    figure_title: str = "PriorNet Spatial-Resolution Unification",
) -> None:
  
    assert model.ndim == 3 and sentinel_2.ndim == 3

    band_sets = [
        ([3, 2, 1], 2.5),   
        ([6, 5, 4], 1.5),  
        ([9, 8, 7], 1.5),   
        ([11, 10, 0], 1.0), 
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(figure_title, fontsize=16)

    for col, (bands, scale) in enumerate(band_sets):
        model_rgb = _prepare_rgb_cube(model, bands, scale)
        sentinel_rgb = _prepare_rgb_cube(sentinel_2, bands, scale)

        axes[0, col].imshow(model_rgb)
        axes[0, col].set_title("PriorNet", fontsize=12)
        axes[0, col].axis("off")

        axes[1, col].imshow(sentinel_rgb)
        axes[1, col].set_title("Input Sentinel-2", fontsize=12)
        axes[1, col].axis("off")

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def sample_signature_indices(
    h: int,
    w: int,
    num_signature_plot: int,
    seed: int | None = None,
) -> np.ndarray:
   
    rng = np.random.default_rng(seed)

    row_indices = rng.permutation(h)[:num_signature_plot]
    col_indices = rng.permutation(w)[:num_signature_plot]

    signature_plot_index = np.stack([row_indices, col_indices], axis=0)
    return signature_plot_index


def plot_spectral_signatures(
    avi_c: np.ndarray,
    ualnet: np.ndarray,
    signature_plot_index: np.ndarray,
    save_path: str | Path,
    gt_label: str = "Reference",
    pred_label: str = "UALNet",
) -> None:
 
    if avi_c.shape != ualnet.shape:
        raise ValueError(f"`avi_c` and `ualnet` must have the same shape, but got {avi_c.shape} and {ualnet.shape}.")

    num_signature_plot = signature_plot_index.shape[1]
    num_rows = round(num_signature_plot / 4)
    num_cols = round(num_signature_plot / num_rows)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows))
    axes = np.array(axes).reshape(-1)

    for i in range(num_signature_plot):
        y = signature_plot_index[0, i]
        x = signature_plot_index[1, i]

        signature_real = avi_c[y, x, :]
        signature_ualnet = ualnet[y, x, :]

        ax = axes[i]
        ax.plot(signature_real, linewidth=2.5, label=gt_label)
        ax.plot(signature_ualnet, linewidth=2.5, label=pred_label)
        ax.set_title(f"Pixel ({y}, {x})", fontsize=10)
        ax.set_xlabel("Band")
        ax.set_ylabel("Value")

    for i in range(num_signature_plot, len(axes)):
        axes[i].axis("off")

    if num_signature_plot > 0:
        axes[0].legend(frameon=False)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def prepare_rgb_image(
    image: np.ndarray,
    band_indices_1based: list[int] | tuple[int, int, int],
    scale: float,
) -> np.ndarray:
 
    band_indices_0based = [b - 1 for b in band_indices_1based]

    if max(band_indices_0based) >= image.shape[2] or min(band_indices_0based) < 0:
        raise ValueError(
            f"Band indices {band_indices_1based} exceed image channel range with shape {image.shape}."
        )

    rgb = image[:, :, band_indices_0based].astype(np.float32) * scale
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def plot_rgb_comparison(
    sentinel_2: np.ndarray,
    avi_c: np.ndarray,
    ualnet: np.ndarray,
    band_for_rgb_1based: list[int] | tuple[int, int, int],
    save_path: str | Path,
) -> None:
   
    sentinel_rgb = prepare_rgb_image(sentinel_2, [4, 3, 2], scale=3.0)
    avi_rgb = prepare_rgb_image(avi_c, band_for_rgb_1based, scale=4.0)
    ualnet_rgb = prepare_rgb_image(ualnet, band_for_rgb_1based, scale=4.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(sentinel_rgb)
    axes[0].set_title("Input Sentinel-2 Data")
    axes[0].axis("off")

    axes[1].imshow(avi_rgb)
    axes[1].set_title("Reference AVIRIS")
    axes[1].axis("off")

    axes[2].imshow(ualnet_rgb)
    axes[2].set_title("UALNet-Generated AVIRIS")
    axes[2].axis("off")

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction(
    avi_c: np.ndarray,
    sentinel_2: np.ndarray,
    ualnet: np.ndarray,
    priornet: np.ndarray,
    save_dir: str | Path,
    num_signature_plot: int = 10,
    band_for_rgb_1based: list[int] | tuple[int, int, int] = (25, 12, 8),
    seed: int | None = 42,
) -> None:
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    h, w, _ = avi_c.shape
    num_signature_plot = min(num_signature_plot, h, w)
    signature_plot_index = sample_signature_indices(h, w, num_signature_plot, seed=seed)

    np.save(save_dir / "signature_plot_index.npy", signature_plot_index)

    plot_spectral_signatures(
        avi_c=avi_c,
        ualnet=ualnet,
        signature_plot_index=signature_plot_index,
        save_path=save_dir / "signature.png",
    )

    plot_rgb_comparison(
        sentinel_2=sentinel_2,
        avi_c=avi_c,
        ualnet=ualnet,
        band_for_rgb_1based=band_for_rgb_1based,
        save_path=save_dir / "ture_color_comparison.png",
    )

    plot_sentinel12_comparison(
        model=priornet,
        sentinel_2=sentinel_2,
        save_path=save_dir / "priornet.png",
    )


