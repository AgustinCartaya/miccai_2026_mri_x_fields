
# import torch
import torch.nn as nn
from typing import Sequence
from monai.networks.blocks import Convolution
from monai.networks.layers import Act

class PatchDiscriminator(nn.Sequential):
    """
    PatchGAN + ACGAN discriminator for MRI domain conditioning.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D etc.)
        num_channels: number of filters in the first convolutional layer (double of the value is taken from then on)
        in_channels: number of input channels
        out_channels: number of output channels in each discriminator
        num_layers_d: number of Convolution layers (Conv + activation + normalisation + [dropout]) in each
            of the discriminators. In each layer, the number of channels are doubled and the spatial size is
            divided by 2.
        kernel_size: kernel size of the convolution layers
        activation: activation layer type
        norm: normalisation type
        bias: introduction of layer bias
        padding: padding to be applied to the convolutional layers
        dropout: proportion of dropout applied, defaults to 0.
        last_conv_kernel_size: kernel size of the last convolutional layer.

        Outputs:
        - patch_logits: (B, 1, H', W', D') realism map
        - class_logits: (B, num_classes) scanner classification
        - features: intermediate feature maps (for feature matching)
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        in_channels: int,
        out_channels: int = 1,
        num_layers_d: int = 3,
        num_classes: int = 5,  # 0.1T, 1.5T, 3T, 5T, 7T
        kernel_size: int = 4,
        activation: str | tuple = (Act.LEAKYRELU, {"negative_slope": 0.2}),
        norm: str | tuple = "INSTANCE",
        bias: bool = False,
        padding: int | Sequence[int] = 1,
        dropout: float | tuple = 0.0,
        last_conv_kernel_size: int | None = None,
    ) -> None: 
        super().__init__()
        if last_conv_kernel_size is None:
            last_conv_kernel_size = kernel_size
        self.num_layers_d = num_layers_d

        # -------------------------
        # Shared convolutional trunk
        # -------------------------
        layers = []

        layers.append(
            Convolution(
                spatial_dims=spatial_dims,
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=num_channels,
                act=activation,
                bias=True,
                norm=None,
                dropout=dropout,
                padding=padding,
                strides=2,
            )
        )

        input_ch = num_channels
        output_ch = num_channels * 2

        for i in range(num_layers_d):
            stride = 1 if i == num_layers_d - 1 else 2

            layers.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    kernel_size=kernel_size,
                    in_channels=input_ch,
                    out_channels=output_ch,
                    act=activation,
                    bias=bias,
                    norm=norm,
                    dropout=dropout,
                    padding=padding,
                    strides=stride,
                )
            )

            input_ch = output_ch
            output_ch *= 2

        self.backbone = nn.Sequential(*layers)

        # -------------------------
        # PatchGAN head (real/fake per patch)
        # -------------------------
        self.patch_head = Convolution(
            spatial_dims=spatial_dims,
            kernel_size=last_conv_kernel_size,
            in_channels=input_ch,
            out_channels=out_channels,
            conv_only=True,
            padding=1,
            strides=1,
        )

        # -------------------------
        # ACGAN head (scanner classification)
        # -------------------------
        self.global_pool = nn.AdaptiveAvgPool3d(1) if spatial_dims == 3 else nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(input_ch, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, num_classes),
        )

        self.apply(self._init_weights)

    # -------------------------
    # forward
    # -------------------------
    def forward(self, x):
        feats = self.backbone(x)

        # Patch realism logits
        patch_logits = self.patch_head(feats)

        # Global classification logits (scanner type)
        pooled = self.global_pool(feats)
        pooled = pooled.view(pooled.shape[0], -1)
        class_logits = self.classifier(pooled)

        return {
            "patch_logits": patch_logits,
            "class_logits": class_logits,
            "features": feats
        }

    # -------------------------
    # initialization
    # -------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)