# Architecture hyperparams for pyiqa's koniq10k MUSIQ checkpoint (musiq_koniq_ckpt-e95806b9.pth).
# Matches pyiqa.archs.musiq_arch.MUSIQ's defaults, which are themselves the paper's ResNet-token +
# multi-scale transformer config. The koniq10k checkpoint's own 1-d regression head is discarded in
# favor of our 5-way MeanOpinionScoreHead (see head.py), matching the vit/cnn/hybrid convention.
patch_size = 32
patch_stride = 32
resnet_token_dim = 64
hidden_size = 384
mlp_dim = 1152
num_heads = 6
head_dim = hidden_size // num_heads
num_layers = 14
dropout_rate = 0.0
attention_dropout_rate = 0.0
abs_value_head_scaledown = 4

# Fixed square resize applied by this repo's SimpleImageProcessor before batching (MUSIQ's own
# multiscale patch extraction otherwise expects variable-resolution input). Coincides with the
# largest entry in longer_side_lengths below, so the "native resolution" scale below overlaps with
# the 384 scale for a freshly-resized image — harmless, just a little redundant compute.
img_size = 384

group_norm_groups = 32
root_group_norm_eps = 1e-6
block_group_norm_eps = 1e-4
layer_norm_eps = 1e-6
std_conv_weight_std_eps = 1e-5

stem_conv_kernel_size = 7
stem_conv_stride = 2
stem_pool_kernel_size = 3
stem_pool_stride = 2
# Two stride-2 ops (conv, then pool) in the stem, so a patch_size x patch_size patch comes out this
# much smaller per side (e.g. 32 -> 8x8) before the linear patch embedding.
stem_spatial_reduction = stem_conv_stride * stem_pool_stride
bottleneck_expansion = 4

embedding_init_std = 0.02
attention_mask_fill_value = -1e3

# [0, 1] -> [-1, 1], matching MUSIQ's original TF preprocessing.
rgb_rescale_mean = 0.5
rgb_rescale_scale = 2.0

spatial_pos_grid_size = 10
longer_side_lengths = [224, 384]
# -1 means "keep every patch from the native-resolution scale, unpadded/untruncated" (rather than
# skipping that scale, which is what pyiqa uses `None` for) — matches pyiqa's MUSIQ default.
max_seq_len_from_original_res = -1
num_scales = len(longer_side_lengths) + 1
