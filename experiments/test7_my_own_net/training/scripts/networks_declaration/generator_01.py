import torch
from torch import nn
from monai.networks.blocks import Convolution, Upsample
from monai.networks.layers.factories import Pool

# import sys
# sys.path.append("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test7_my_own_net/training/scripts/networks_declaration")
# # set current directoyr as python path


from .spatialattention import SpatialAttentionBlock

from collections.abc import Sequence

from monai.utils import ensure_tuple_rep

from monai.utils.type_conversion import convert_to_tensor

# from monai.networks.layers import Act, get_pool_layer




### -------------------------------- Commons





class DiffusionUnetDownsample(nn.Module):
    """
    Downsampling layer.

    Args:
        spatial_dims: number of spatial dimensions.
        num_channels: number of input channels.
        use_conv: if True uses Convolution instead of Pool average to perform downsampling. In case that use_conv is
            False, the number of output channels must be the same as the number of input channels.
        out_channels: number of output channels.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension.
    """

    def __init__(
        self, spatial_dims: int, num_channels: int, use_conv: bool, out_channels: int | None = None, padding: int = 1
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        if use_conv:
            self.op = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.num_channels,
                out_channels=self.out_channels,
                strides=2,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            if self.num_channels != self.out_channels:
                raise ValueError("num_channels and out_channels must be equal when use_conv=False")
            self.op = Pool[Pool.AVG, spatial_dims](kernel_size=2, stride=2)

    # def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # del emb
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Input number of channels ({x.shape[1]}) is not equal to expected number of channels "
                f"({self.num_channels})"
            )
        output: torch.Tensor = self.op(x)
        return output


class WrappedUpsample(Upsample):
    """
    Wraps MONAI upsample block to allow for calling with timestep embeddings.
    """

    # def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # del emb
        upsampled: torch.Tensor = super().forward(x)
        return upsampled



class DiffusionUNetResnetBlock(nn.Module):
    """
    Residual block with timestep conditioning.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        # temb_channels: number of timestep embedding  channels.
        out_channels: number of output channels.
        up: if True, performs upsampling.
        down: if True, performs downsampling.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        # temb_channels: int,
        out_channels: int | None = None,
        up: bool = False,
        down: bool = False,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = in_channels
        # self.emb_channels = temb_channels
        self.out_channels = out_channels or in_channels
        self.up = up
        self.down = down

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.nonlinearity = nn.SiLU()
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = WrappedUpsample(
                spatial_dims=spatial_dims,
                mode="nontrainable",
                in_channels=in_channels,
                out_channels=in_channels,
                interp_mode="nearest",
                scale_factor=2.0,
                align_corners=None,
            )
        elif down:
            self.downsample = DiffusionUnetDownsample(spatial_dims, in_channels, use_conv=False)

        # self.time_emb_proj = nn.Linear(temb_channels, self.out_channels)
        self.cond_proj = None
        if cond_dim is not None:
            self.cond_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, 2 * self.out_channels)
            )

        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.out_channels, eps=norm_eps, affine=True)
        self.conv2 = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        self.skip_connection: nn.Module
        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )

    # def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)

        if self.upsample is not None:
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)

        # if self.spatial_dims == 2:
        #     temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None]
        # else:
        #     temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None, None]
        # h = h + temb

        h = self.norm2(h)

        if self.cond_proj is not None and cond is not None:

            gamma_beta = self.cond_proj(cond)

            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)

            if self.spatial_dims == 2:
                gamma = gamma[:, :, None, None]
                beta = beta[:, :, None, None]
            else:
                gamma = gamma[:, :, None, None, None]
                beta = beta[:, :, None, None, None]

            h = h * (1 + gamma) + beta


        h = self.nonlinearity(h)
        h = self.conv2(h)
        output: torch.Tensor = self.skip_connection(x) + h
        return output




### -------------------------------- Down Block


class DownBlock(nn.Module):
    """
    Unet's down block containing resnet and downsamplers blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        # temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_downsample: if True add downsample block.
        resblock_updown: if True use residual blocks for downsampling.
        downsample_padding: padding used in the downsampling block.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        # temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    cond_dim=cond_dim,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsampler: nn.Module | None
            if resblock_updown:
                self.downsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    # temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = DiffusionUnetDownsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    # def forward(
    #     self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    # ) -> tuple[torch.Tensor, list[torch.Tensor]]:

    def forward(self, hidden_states: torch.Tensor, cond: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # del context
        output_states = []

        for resnet in self.resnets:
            # hidden_states = resnet(hidden_states, temb)
            hidden_states = resnet(hidden_states, cond)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            # hidden_states = self.downsampler(hidden_states, temb)
            hidden_states = self.downsampler(hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


### -------------------------------- Middle block



class AttnMidBlock(nn.Module):
    """
    Unet's mid block containing resnet and self-attention blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        # temb_channels: number of timestep embedding channels.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        num_head_channels: number of channels in each attention head.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        # temb_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.resnet_1 = DiffusionUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            # temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cond_dim=cond_dim
        )
        self.attention = SpatialAttentionBlock(
            spatial_dims=spatial_dims,
            num_channels=in_channels,
            num_head_channels=num_head_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )

        self.resnet_2 = DiffusionUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            # temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cond_dim=cond_dim
        )

    # def forward(
    #     self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    # ) -> torch.Tensor:

    def forward(self, hidden_states: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        # del context
        # hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.resnet_1(hidden_states, cond)
        hidden_states = self.attention(hidden_states).contiguous()
        # hidden_states = self.resnet_2(hidden_states, temb)
        hidden_states = self.resnet_2(hidden_states, cond)

        return hidden_states


### -------------------------------- Up Block


class UpBlock(nn.Module):
    """
    Unet's up block containing resnet and upsamplers blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        prev_output_channel: number of channels from residual connection.
        out_channels: number of output channels.
        # temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_upsample: if True add downsample block.
        resblock_updown: if True use residual blocks for upsampling.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        # temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown
        resnets = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    # temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    cond_dim=cond_dim
                )
            )

        self.resnets = nn.ModuleList(resnets)

        self.upsampler: nn.Module | None
        if add_upsample:
            if resblock_updown:
                self.upsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    # temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                post_conv = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
                self.upsampler = WrappedUpsample(
                    spatial_dims=spatial_dims,
                    mode="nontrainable",
                    in_channels=out_channels,
                    out_channels=out_channels,
                    interp_mode="nearest",
                    scale_factor=2.0,
                    post_conv=post_conv,
                    align_corners=None,
                )

        else:
            self.upsampler = None

    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     res_hidden_states_list: list[torch.Tensor],
    #     temb: torch.Tensor,
    #     context: torch.Tensor | None = None,
    # ) -> torch.Tensor:
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # del context
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            # hidden_states = resnet(hidden_states, temb)
            hidden_states = resnet(hidden_states, cond)

        if self.upsampler is not None:
            # hidden_states = self.upsampler(hidden_states, temb)
            hidden_states = self.upsampler(hidden_states)

        return hidden_states




### -------------------------------- Full network



class MultiResolutionGenerator(nn.Module):
    """
    U-Net network with timestep embedding and attention mechanisms for conditioning based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
    and Pinaya et al. "Brain Imaging Generation with Latent Diffusion Models" https://arxiv.org/abs/2209.07162

    Args:
        spatial_dims: Number of spatial dimensions.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_res_blocks: Number of residual blocks (see ResnetBlock) per level. Can be a single integer or a sequence of integers.
        num_channels: Tuple of block output channels.
        attention_levels: List of levels to add attention.
        norm_num_groups: Number of groups for the normalization.
        norm_eps: Epsilon for the normalization.
        resblock_updown: If True, use residual blocks for up/downsampling.
        upcast_attention: If True, upcast attention operations to full precision.
        include_fc: whether to include the final linear layer. Default to False.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: If True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        include_fc: bool = False,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        nb_resolutions: int = 5,
        resolution_emb_channels: int | None = None,
    ) -> None:
        # print("instantiating DiffusionModelUNetMaisi")
        super().__init__()


        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in num_channels):
            raise ValueError(
                f"DiffusionModelUNetMaisi expects all num_channels being multiple of norm_num_groups, "
                f"but get num_channels: {num_channels} and norm_num_groups: {norm_num_groups}"
            )

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )

        if use_flash_attention is True and not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. Flash attention is only available for GPU."
            )

        self.in_channels = in_channels
        self.block_out_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        # conditioning dimension
        self.field_embedding = None
        if resolution_emb_channels is not None:
            self.field_embedding = nn.Embedding(nb_resolutions, resolution_emb_channels)

        # input
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        # down
        self.down_blocks = nn.ModuleList([])
        output_channel = num_channels[0]
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1
            down_block = DownBlock(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
                resblock_updown=resblock_updown,
                cond_dim=resolution_emb_channels,
            )

            self.down_blocks.append(down_block)

        # mid
        self.middle_block = AttnMidBlock(
            spatial_dims=spatial_dims,
            in_channels=num_channels[-1],
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_head_channels=1,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
            cond_dim=resolution_emb_channels
        )

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(num_channels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(num_channels) - 1)]

            is_final_block = i == len(num_channels) - 1

            up_block = UpBlock(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                num_res_blocks=reversed_num_res_blocks[i] + 1,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_upsample=not is_final_block,
                resblock_updown=resblock_updown,
                # cond_dim=resolution_emb_channels # No conditioning for up blocks, as we only condition on the down blocks and mid block
            )

            self.up_blocks.append(up_block)



        # resolution_residuals
        mult_residuals = 1
        self.resolution_residuals_list = nn.ModuleList([])
        res_residual_in_channels = num_channels[0]

        for i in range(nb_resolutions-1):
            res_residual_out_channels = int(res_residual_in_channels * mult_residuals)
            # __norm_num_groups = res_residual_out_channels//2
            out =  DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=res_residual_in_channels,
                    out_channels=res_residual_out_channels,
                    # temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            self.resolution_residuals_list.append(out)

            res_residual_in_channels = res_residual_out_channels

        # out # 
        self.out_list = nn.ModuleList([])
        out_in_channels = num_channels[0]
        for i in range(nb_resolutions):
            out = nn.Sequential(
                nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_in_channels, eps=norm_eps, affine=True),
                nn.SiLU(),
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=out_in_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            )
            self.out_list.append(out)
            out_in_channels = int(out_in_channels * mult_residuals)

    def _apply_down_blocks(self, h, cond=None):
        down_block_res_samples: list[torch.Tensor] = [h]
        for i, downsample_block in enumerate(self.down_blocks):
            h, res_samples = downsample_block(hidden_states=h, cond=cond)
            down_block_res_samples.extend(res_samples)


        return h, down_block_res_samples




    def _apply_up_blocks(self, h, down_block_res_samples):
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples)

        return h
    

    def _apply_resolution_residuals(self, h):
        resolution_residuals: list[torch.Tensor] = [h]
        for res_block in self.resolution_residuals_list:
            h = res_block(h)
            resolution_residuals.append(h)
        return resolution_residuals

    def _apply_out_blocks(self, resolution_residuals):
        out_samples: list[torch.Tensor] = []
        for i, out_block in enumerate(self.out_list):
            h = resolution_residuals[i]
            h = out_block(h)
            out_samples.append(convert_to_tensor(h))
        return out_samples
    
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the UNet model.

        Args:
            x: Input tensor of shape (N, C, SpatialDims).
        Returns:
            A tensor representing the output of the UNet model.
        """

        # condition embedding
        cond_emb = None
        if self.field_embedding is not None and cond is not None:
            cond_emb = self.field_embedding(cond)

        h = self.conv_in(x)
        h, _updated_down_block_res_samples = self._apply_down_blocks(h, cond_emb)
        h = self.middle_block(h, cond_emb)
        h = self._apply_up_blocks(h, _updated_down_block_res_samples)

        resolution_residuals = self._apply_resolution_residuals(h)
        outs = self._apply_out_blocks(resolution_residuals)
        return outs

