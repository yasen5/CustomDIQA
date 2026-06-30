import math

dropout = 0.2
drop_connect_rate = 0.2
batch_norm_momentum = 0.01
batch_norm_epsilon = 0.001
se_ratio = 0.25
depth_divisor = 8
abs_value_head_scaledown = 4
img_size = 224

stem_channels = 32
head_channels = 1280
width_multiplier = 1.0
depth_multiplier = 1.0

# (expand_ratio, out_channels, num_repeats, stride, kernel_size)
base_stage_settings = [
    (1, 16, 1, 1, 3),
    (6, 24, 2, 2, 3),
    (6, 40, 2, 2, 5),
    (6, 80, 3, 2, 3),
    (6, 112, 3, 1, 5),
    (6, 192, 4, 2, 5),
    (6, 320, 1, 1, 3),
]


def round_filters(channels: int) -> int:
    scaled_channels = channels * width_multiplier
    new_channels = max(depth_divisor, int(scaled_channels + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_channels < 0.9 * scaled_channels:
        new_channels += depth_divisor
    return int(new_channels)


def round_repeats(repeats: int) -> int:
    return int(math.ceil(depth_multiplier * repeats))


def scaled_stage_settings():
    return [
        (expand_ratio, round_filters(out_channels), round_repeats(num_repeats), stride, kernel_size)
        for expand_ratio, out_channels, num_repeats, stride, kernel_size in base_stage_settings
    ]
