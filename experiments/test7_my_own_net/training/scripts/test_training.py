import torch

from networks_declaration.generator_01 import MultiResolutionGenerator
from networks_declaration.discriminator_01 import PatchDiscriminator
from monai.networks.layers import Act, get_pool_layer


device_name = f"cuda:3"
device = torch.device(device_name)

spatial_dims = 3
in_channels = 4
out_channels = 4
num_res_blocks = 2
num_channels = [32,64,128,256] #[8,16,32,64]
norm_num_groups = 8
norm_eps = 1e-6
resblock_updown = False
include_fc = False
use_flash_attention = True

generator = MultiResolutionGenerator(
    spatial_dims=spatial_dims,
    in_channels=in_channels,
    out_channels=out_channels,
    num_res_blocks=num_res_blocks,
    num_channels=num_channels,
    norm_num_groups=norm_num_groups,
    norm_eps=norm_eps,
    resblock_updown=resblock_updown,
    include_fc=include_fc,
    use_flash_attention=use_flash_attention,
    nb_resolutions=5,
    resolution_emb_channels=64
)

discriminator = PatchDiscriminator(
    spatial_dims=spatial_dims,
    num_channels=32,
    in_channels=out_channels,
    out_channels=out_channels,
    num_layers_d=3,
    kernel_size=4,
    activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
    norm="INSTANCE",
    bias=False,
    padding=1,
    dropout=0.0,
    last_conv_kernel_size=4,
)


print("Gen number of parameters: ", sum(p.numel() for p in generator.parameters() if p.requires_grad))
print("Disc number of parameters: ", sum(p.numel() for p in discriminator.parameters() if p.requires_grad))

# input = torch.randn(1, in_channels, 64,64,64)  # Example input tensor with shape (N, C, D, H, W)
input = torch.randn(6, in_channels, 96,112,96)  # Example input tensor with shape (N, C, D, H, W)
input = input.to(device)
generator = generator.to(device)
discriminator = discriminator.to(device)
output_g = generator(input)  # Forward pass through the generator model
output_d = discriminator(input)

print("Gen Output shape: ", output_g.shape)  # Print the shape of the output tensor
print("Gen Output shape: ", [o.shape for o in output_g])  # Print the shape of the output tensor
print("Disc Output shape: ", [o.shape for o in output_d["patch_logits"]])  # Print the shape of the output tensor
print("Disc Output shape: ", [o.shape for o in output_d["class_logits"]])  # Print the shape of the output tensor
print("Disc Output shape: ", [o.shape for o in output_d["features"]])  # Print the shape of the output tensor
# print("Disc Output shape: ", [o.shape for o in output_d])  # Print the shape of the output tensor
# print(generator)
# print(discriminator)