import torch
import torch.nn as nn
from einops import rearrange

from .module import MRF, CAM, ChannelAttention


class PriorNet(nn.Module):
    """
    PriorNet for Sentinel-2 spatial resolution unification and prior spectral matrix estimation.

    Args:
        in_channels (int): Number of input channels (wich is 12 for typical Sentinal-2 data).
        out_channels (int): Number of output channels (wich is 12 for typical Sentinal-2 data).
        feat_dim (int): Internal feature dimension.
        hyperspectral_channels (int): Number of hyperspectral channels used
            in the prior estimation branch.
        num_stages (int): Number of iterative refinement stages.
        upsample_scale (int): Upsampling factor.
        eps (float): Small value to avoid division by zero.

    Inputs:
        x (torch.Tensor): Input tensor of shape (B, C, H, W),
            where C == in_channels.

    Returns:
        sen (torch.Tensor): Resolution-unified Sentinel-2 output tensor of shape
            (B, out_channels, 2H, 2W) if upsample_scale=2.
        SPM (torch.Tensor): Spectral prior matrix computed from the
            prior branch, with shape (B, N, N), where N (i.e., hyperspectral_channels) depends on the
            spectral resolution of the target hyperspectral image.
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 12,
        feat_dim: int = 12,
        hyperspectral_channels: int = 186,
        num_stages: int = 3,
        upsample_scale: int = 2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, but got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, but got {out_channels}.")
        if feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive, but got {feat_dim}.")
        if hyperspectral_channels <= 0:
            raise ValueError(
                f"hyperspectral_channels must be positive, but got {hyperspectral_channels}."
            )
        if num_stages <= 0:
            raise ValueError(f"num_stages must be positive, but got {num_stages}.")
        if upsample_scale <= 0:
            raise ValueError(
                f"upsample_scale must be positive, but got {upsample_scale}."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_dim = feat_dim
        self.hyperspectral_channels = hyperspectral_channels
        self.num_stages = num_stages
        self.upsample_scale = upsample_scale
        self.eps = eps

        # ------------------------------------------------------------------
        # Upsampling branch
        # ------------------------------------------------------------------
        self.upsample = nn.Upsample(
            scale_factor=upsample_scale,
            mode="bicubic",
            align_corners=True,
        )

        # ------------------------------------------------------------------
        # Spatial Dimension Alignment
        # ------------------------------------------------------------------
        self.SDA_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * 3,
                kernel_size=3,
                padding=1,
            ),
            MRF(dim=in_channels * 3),
            nn.Conv2d(
                in_channels=in_channels * 3,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        self.input_proj = nn.Conv2d(in_channels, feat_dim, kernel_size=1, padding=0)

        # ------------------------------------------------------------------
        # Multiscale encoding branch
        # ------------------------------------------------------------------
        self.M_Enconder = nn.Sequential(
            MRF(dim=feat_dim),
            MRF(dim=feat_dim),
            MRF(dim=feat_dim),
            MRF(dim=feat_dim),
        )

        self.Bottleneck = nn.ModuleList()
        for _ in range(num_stages):
            self.Bottleneck.append(
                nn.ModuleList(
                    [
                        MRF(dim=feat_dim),
                        CAM(channel=feat_dim),
                    ]
                )
            )
        
        # ------------------------------------------------------------------
        # Decoding branch
        # ------------------------------------------------------------------

        fusion_in_channels = feat_dim * (num_stages + 1) + in_channels
        self.fusion = nn.Conv2d(
            fusion_in_channels, feat_dim, kernel_size=3, padding=1
        )

        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(feat_dim, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.skip_head = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        # ------------------------------------------------------------------
        # Prior estimation branch
        # ------------------------------------------------------------------
        reduced_channels = hyperspectral_channels // 4
        if reduced_channels <= 0:
            raise ValueError(
                "hyperspectral_channels // 4 must be >= 1. "
                f"Got hyperspectral_channels={hyperspectral_channels}."
            )

        self.initial_prior = nn.Sequential(
            ChannelAttention(endmember=in_channels),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.Conv2d(
                reduced_channels,
                reduced_channels,
                kernel_size=2,
                stride=1,
                padding=1,
                bias=False,
                groups=reduced_channels,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ChannelAttention(endmember=reduced_channels),
            nn.Conv2d(reduced_channels, hyperspectral_channels, kernel_size=1),
            nn.Conv2d(
                hyperspectral_channels,
                hyperspectral_channels,
                kernel_size=2,
                stride=1,
                padding=1,
                bias=False,
                groups=hyperspectral_channels,
            ),
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor shape and type."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected x to be torch.Tensor, but got {type(x)}.")

        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape (B, C, H, W), but got tensor with shape {tuple(x.shape)}."
            )

        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels, "
                f"but got {x.size(1)} channels."
            )

        if x.size(-1) < 1 or x.size(-2) < 1:
            raise ValueError(
                f"Input spatial size must be positive, but got {tuple(x.shape)}."
            )

    def _compute_prior(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute prior features and flatten the spatial dimensions.

        Returns:
            prior (torch.Tensor): Shape (B, C_prior, H_prior * W_prior)
        """
        prior = self.initial_prior(x)
        prior = rearrange(prior, "b c h w -> b c (h w)")
        return prior

    def _normalize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize input by channel-wise spatial mean after upsampling.

        Returns:
            x_up_norm (torch.Tensor): Normalized upsampled tensor.
            mean (torch.Tensor): Spatial mean tensor of shape (B, C, 1, 1).
        """
        mean = torch.mean(x, dim=(-2, -1), keepdim=True)
        mean = mean.clamp_min(self.eps)

        x_up = self.upsample(x)
        x_up_norm = x_up / mean
        return x_up_norm, mean

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
      
        self._validate_input(x)

        # Prior branch
        prior = self._compute_prior(x)

        # Main branch
        x_up, mean = self._normalize_input(x)
        x_up = self.SDA_block(x_up) + x_up

        feat = self.input_proj(x_up)
        feat = self.M_Enconder(feat)

        feat_list = [feat]
        current_feat = feat

        for local_block, attention_block in self.Bottleneck:
            current_feat = attention_block(local_block(current_feat))
            feat_list.append(current_feat)

        fused_feat = torch.cat(feat_list + [x_up], dim=1)
        fused_feat = self.fusion(fused_feat)

        output = self.reconstruction_head(fused_feat) + self.skip_head(x_up)

        # Recover scale
        sen = output * mean

        # Prior  matrix
        SPM = prior @ prior.permute(0, 2, 1)

        return sen, SPM

if __name__ == "__main__":

    # -------------------------------------------------
    # Device
    # -------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------
    BATCH_SIZE = 2
    IN_CHANNELS = 12
    HEIGHT = 128
    WIDTH = 128

    # -------------------------------------------------
    # Initialize Model
    # -------------------------------------------------
    model = PriorNet(
        in_channels=IN_CHANNELS,
        out_channels=12,
        feat_dim=12,
        hyperspectral_channels=186,
        num_stages=3,
    ).to(device)

    model.eval()

    print("\nModel successfully initialized.\n")

    # -------------------------------------------------
    # Generate Random Input
    # -------------------------------------------------
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH).to(device)

    print(f"Input shape : {x.shape}")

    # -------------------------------------------------
    # Forward Pass
    # -------------------------------------------------
    with torch.no_grad():
        sen, prior_gram = model(x)

    # -------------------------------------------------
    # Output Information
    # -------------------------------------------------
    print("\nForward pass successful!\n")

    print(f"Output (Resolution-unified Sentinel-2) shape : {sen.shape}")
    print(f"Spectral Prior matrix shape   : {prior_gram.shape}")

    # -------------------------------------------------
    # Parameter Count
    # -------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel Parameters")
    print("-" * 60)
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")

    print("\nTest completed successfully.")
