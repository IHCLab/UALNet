import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class Discriminator(nn.Module):
    """
    Discriminator network for unfolding adversarial learning

    Args:
        in_channels (int): Number of input spectral channels.
        hidden_channels (int): Base hidden channel dimension.
    """

    def __init__(
        self,
        in_channels: int = 186,
        hidden_channels: int = 15,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # ------------------------------------------------------------------
        # Feature encoder
        # ------------------------------------------------------------------
        self.encoder_stage1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            GELU(),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            GELU(),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            GELU(),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            GELU(),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1),
            GELU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )

        # ------------------------------------------------------------------
        # Spectral / spatial classifiers
        # ------------------------------------------------------------------
        self.spatial_classifier = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, kernel_size=3, padding=1),
            GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels * 8, in_channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        self.spectral_classifier = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1),
            GELU(),
            nn.Conv2d(hidden_channels * 4, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """

        feat = self.encoder_stage1(x)
        feat = self.encoder_stage2(feat)

        spatial_score = self.spatial_classifier(self.encoder_stage3(feat))  # (B, C, 1, 1)
        spectral_score = self.spectral_classifier(feat)                     # (B, 1, H, W)

        output = spatial_score * spectral_score
        return output


if __name__ == "__main__":
    print("=" * 60)
    print("Discriminator Quick Test")
    print("=" * 60)

    batch_size = 3
    in_channels = 186
    height = 252
    width = 252

    x = torch.randn(batch_size, in_channels, height, width)

    model = Discriminator(in_channels=in_channels, hidden_channels=15)
    model.eval()

    with torch.no_grad():
        output = model(x)

    print(f"Input shape  : {tuple(x.shape)}")
    print(f"Output shape : {tuple(output.shape)}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")
    print("Quick test finished successfully.")
