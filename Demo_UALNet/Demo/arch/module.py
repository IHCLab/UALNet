import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class ChannelAttention(nn.Module):
    def __init__(self, endmember: int, pooling_size: int = 1):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(pooling_size)
        self.max_pool = nn.AdaptiveMaxPool2d(pooling_size)
        self.sigmoid = nn.Sigmoid()

        self.layer = nn.Sequential(
            nn.Conv2d(
                endmember,
                endmember,
                kernel_size=1,
                groups=endmember,
            ),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(
                endmember,
                endmember,
                kernel_size=1,
                groups=endmember,
            ),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(
                endmember,
                endmember,
                kernel_size=1,
                groups=endmember,
            ),
        )

    def forward(self, x):
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)

        attention = self.layer(x_avg) + self.layer(x_max)
        attention = self.sigmoid(attention)

        return attention * x


class SpatialAttention(nn.Module):
    def __init__(self, endmember: int, depth: int = 1, kernel_size: int = 3):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(endmember, 1, kernel_size, padding=1),
                        nn.BatchNorm2d(1),
                        nn.Sigmoid(),
                    ]
                )
            )

    def forward(self, x):
        x_conv = x
        for conv, bn, sig in self.layers:
            x_conv = conv(x_conv)
            x_conv = bn(x_conv)
            x_conv = sig(x_conv)

        return x_conv * x


class CAM(nn.Module):
    def __init__(self, channel: int):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(1):
            self.layers.append(
                nn.ModuleList(
                    [
                        ChannelAttention(channel),
                        SpatialAttention(channel),
                    ]
                )
            )

    def forward(self, x):
        identity = x

        for spectral, spatial in self.layers:
            x = spatial(spectral(x))

        return 0.4 * identity + 0.6 * x


class MRF(nn.Module):
    def __init__(self, dim: int, bias: bool = True):
        super().__init__()

        hidden_features = 12

        self.project_in = nn.Conv2d(
            dim,
            hidden_features,
            kernel_size=1,
            bias=bias,
        )

        self.dwconv3x3 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=bias,
        )
        self.dwconv5x5 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=bias,
        )
        self.dwconv7x7 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=bias,
        )

        self.dwconv3x3_2 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=bias,
        )
        self.dwconv5x5_2 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=hidden_features,
            bias=bias,
        )
        self.dwconv7x7_2 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=hidden_features,
            bias=bias,
        )

        self.relu3 = nn.Sequential(GELU(), nn.BatchNorm2d(hidden_features))
        self.relu5 = nn.Sequential(GELU(), nn.BatchNorm2d(hidden_features))
        self.relu7 = nn.Sequential(GELU(), nn.BatchNorm2d(hidden_features))

        self.relu3_1 = nn.Sequential(GELU(), nn.BatchNorm2d(hidden_features))
        self.relu5_1 = nn.Sequential(GELU(), nn.BatchNorm2d(hidden_features))
        self.relu7_1 = nn.Sequential(GELU(), nn.BatchNorm2d(hidden_features))

        self.project_out = nn.Sequential(
            nn.Conv2d(
                hidden_features * 3,
                dim,
                kernel_size=1,
                padding=0,
                bias=bias,
            )
        )

    def forward(self, x):
        identity = x

        x = self.project_in(x)

        x3 = self.relu3(self.dwconv3x3(x))
        x5 = self.relu5(self.dwconv5x5(x3))
        x7 = self.relu7(self.dwconv7x7(x5))

        x1_3, x2_3, x3_3 = x3.chunk(3, dim=1)
        x1_5, x2_5, x3_5 = x5.chunk(3, dim=1)
        x1_7, x2_7, x3_7 = x7.chunk(3, dim=1)

        x1 = torch.cat([x1_3, x1_5, x1_7], dim=1)
        x2 = torch.cat([x2_3, x2_5, x2_7], dim=1)
        x3 = torch.cat([x3_3, x3_5, x3_7], dim=1)

        x1 = self.dwconv3x3_2(x1)
        x2 = self.dwconv5x5_2(x2)
        x3 = self.dwconv7x7_2(x3)

        x1 = self.relu3_1(x1)
        x2 = self.relu5_1(x2)
        x3 = self.relu7_1(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.project_out(x)

        return x + identity


class DFUSBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.input_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.up_conv1 = nn.Conv2d(
            in_channels=128,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.up_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=16,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.down_conv1 = nn.Conv2d(
            in_channels=128,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.down_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=16,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.fusion_conv = nn.Conv2d(
            in_channels=96,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C + 32, H, W).
        """
        feat = self.relu(self.input_proj(x))

        feat_up1 = self.relu(self.up_conv1(feat))
        feat_up2 = self.relu(self.up_conv2(feat_up1))

        feat_down1 = self.relu(self.down_conv1(feat))
        feat_down2 = self.relu(self.down_conv2(feat_down1))

        fused_feat = torch.cat(
            [feat_up1, feat_up2, feat_down1, feat_down2],
            dim=1,
        )
        fused_feat = self.relu(self.fusion_conv(fused_feat))

        out = torch.cat([x, fused_feat], dim=1)
        return out


class DDFN(nn.Module):
    def __init__(self, in_channels: int, num_blocks: int = 10) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_blocks = num_blocks

        self.up_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.up_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.down_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.down_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        dfus_blocks = [
            DFUSBlock(in_channels=128 + 32 * i)
            for i in range(num_blocks)
        ]
        self.dfus_blocks = nn.Sequential(*dfus_blocks)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, 128 + 32 * num_blocks, H, W).
        """
        feat_up1 = self.relu(self.up_conv1(x))
        feat_up2 = self.relu(self.up_conv2(feat_up1))

        feat_down1 = self.relu(self.down_conv1(x))
        feat_down2 = self.relu(self.down_conv2(feat_down1))

        fused_feat = torch.cat(
            [feat_up1, feat_up2, feat_down1, feat_down2],
            dim=1,
        )

        out = self.dfus_blocks(fused_feat)
        return out

"""
The InitialBlock code is modified from the implementation:
https://github.com/caiyuanhao1998/MST-plus-plus
We thank the original authors for making their code publicly available. 
"""
class InitialBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 186,
        num_blocks: int = 10,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        self.ddfn = DDFN(
            in_channels=in_channels,
            num_blocks=num_blocks,
        )

        self.output_proj = nn.Conv2d(
            in_channels=128 + 32 * num_blocks,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, out_channels, H, W).
        """
        feat = self.ddfn(x)
        out = self.output_proj(feat)
        return out