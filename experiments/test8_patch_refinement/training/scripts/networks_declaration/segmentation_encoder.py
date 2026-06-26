


from monai.networks.blocks import ResidualUnit, Convolution
import torch.nn as nn



class SegmentationEncoder(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        num_channels,
        num_res_blocks=2,
    ):
        super().__init__()

        assert spatial_dims == 3

        self.num_levels = len(num_channels)

        # stem
        self.stem = Convolution(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_channels[0],
            kernel_size=1,
            strides=1,
            act="PRELU",
            norm="INSTANCE",
        )

        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        output_channel = num_channels[0]

        for i in range(self.num_levels):
            input_channel = output_channel
            output_channel = num_channels[i]
            # ch = num_channels[i]

            self.blocks.append(
                ResidualUnit(
                    spatial_dims=3,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    act="PRELU",
                    norm="INSTANCE",
                    subunits=num_res_blocks,
                )
            )

            if i < self.num_levels - 1:
                self.downsamples.append(
                    Convolution(
                        spatial_dims=3,
                        in_channels=output_channel,
                        out_channels=output_channel,
                        kernel_size=3,
                        strides=2,
                        act="PRELU",
                        norm="INSTANCE",
                    )
                )

        # mid block (correctly defined)
        self.mid_block = ResidualUnit(
            spatial_dims=3,
            in_channels=num_channels[-1],
            out_channels=num_channels[-1],
            act="PRELU",
            norm="INSTANCE",
            subunits=num_res_blocks,
        )

    def forward(self, x):

        # if the segmentation is not one-hot, we can one-hot encode it here
        if x.shape[1] != self.stem.in_channels:
            # assume x is (B, D, H, W) with integer labels, and convert to one-hot
            # x = nn.functional.one_hot(x.long(), num_classes=self.stem.in_channels).squeeze(1)  # (B, D, H, W, C)
            # print(f"One-hot encoding segmentation input: original shape {x.shape}, new shape {x.shape}")
            # x = x.permute(0, 4, 1, 2, 3).float() # (B, C, D, H, W)
            raise ValueError(f"Expected segmentation input with {self.stem.in_channels} channels, but got {x.shape[1]}. Please one-hot encode the segmentation input before passing it to the model.")

        features = {}

        x = self.stem(x)

        down_features = []

        for i in range(self.num_levels):
            # down_features.append(x) # option 2: add before the block, so we have the features before the residual blocks
            
            x = self.blocks[i](x)

            if i < self.num_levels - 1:
                x = self.downsamples[i](x)

            down_features.append(x) # option 1: add here for one more, put before, and after the for in _apply_down_blocks

        # down_features.append(x) # option 2: add here for one more, put before, and after the for in _apply_down_blocks


        features["down"] = down_features
        features["mid"] = self.mid_block(x)

        return features
    



# from monai.networks.blocks import ResidualUnit, Convolution
# import torch.nn as nn

# class MaskEncoder(nn.Module):
#     def __init__(
#         self,
#         spatial_dims,
#         in_channels,
#         num_channels,
#         num_res_blocks=2,
#     ):
#         super().__init__()

#         assert spatial_dims == 3, "Only 3D supported for now"

#         self.num_levels = len(num_channels)

#         # stem (match first UNet channel)
#         # self.stem = nn.Conv3d(in_channels, num_channels[0], kernel_size=1)

#         self.stem = Convolution(
#             spatial_dims=3,
#             in_channels=in_channels,
#             out_channels=num_channels[0],
#             strides=1,
#             kernel_size=1,
#             act="PRELU",
#             norm="INSTANCE",
#         )

#         # encoder layers
#         self.blocks = nn.ModuleList()
#         self.downsamples = nn.ModuleList()

#         for i in range(self.num_levels):

#             ch = num_channels[i]

#             # residual blocks per level
#             res_blocks = ResidualUnit(
#                     spatial_dims=3,
#                     in_channels=ch,
#                     out_channels=ch,
#                     subunits=num_res_blocks,
#                     act="PRELU",
#                     norm="INSTANCE",
#                 )
            
#             self.blocks.append(res_blocks)

#             # downsample (except last level)
#             if i != self.num_levels - 1:
#                 self.downsamples.append(
#                     Convolution(
#                         spatial_dims=3,
#                         in_channels=ch,
#                         out_channels=num_channels[i + 1],
#                         strides=2,
#                         kernel_size=3,
#                         act="PRELU",
#                         norm="INSTANCE",
#                     )
#                 )
            

#     def forward(self, x):

#         features = []

#         x = self.stem(x)

#         for i in range(self.num_levels):

#             x = self.blocks[i](x)
#             features.append(x)

#             if i < self.num_levels - 1:
#                 x = self.downsamples[i](x)

#         return {
#             f"f{i}": features[i] for i in range(len(features))
#         }
    