# %%
import cv2 as cv
import tifffile as tf
import matplotlib.pyplot as plt
import numpy as np
from util import global_registration, create_image_block_stack, blocks_registration, calculate_MI, calculate_MSD, calculate_NCC, calculate_NMI
import warnings
warnings.filterwarnings("ignore")
print("OK!")
# %%


class args(object):
    float_img_path = 'float_img.tif'
    refer_img_path = 'refer_img.tif'
    save_img_path = 'save_img.tif'

    # method = "SIFT","ORB","KAZE","AKAZE","BRISK","nCCM"
    global_registration_method = 'SIFT'

    pyramid_threshold = 0.25  # (0,1)
    pyramid_layer = 16  # (1,inf)
    pyramid_gaussian_range = 5
    pyramid_interp_method = cv.INTER_CUBIC
    pyramid_max_try = 5  # (1,inf)

    show_global_fig = False
    show_pyramid_fig = False


para = args()
# %%
#!############################################################################################
#!############################################################################################
#! 主程序开始
new = tf.imread(para.float_img_path).astype(np.uint8)
refer = tf.imread(para.refer_img_path).astype(np.uint8)
# %%
global_method = para.global_registration_method
target = refer
warp_img = global_registration(
    new, target, global_method, para.show_global_fig)
first_warp_img = warp_img
# %%
block_row_ini = block_row = 1
block_col_ini = block_col = 1
target_block_stack = create_image_block_stack(target, block_row, block_col)
# %%
threshold = para.pyramid_threshold
r = para.pyramid_gaussian_range
layer = para.pyramid_layer
layer_state = np.ones(layer)
warp_img_layer = np.zeros([warp_img.shape[0], warp_img.shape[1], layer])
target_img_layer = np.zeros([target.shape[0], target.shape[1], layer])
target = target.astype(np.float64)
warp_img = warp_img.astype(np.float64)
refer = refer.astype(np.float64)
method = para.pyramid_interp_method
# %%
for i in range(layer):
    count = 0
    rtmap = np.zeros_like(target)
    ctmap = np.zeros_like(target)
    warp_img_layer[:, :, i] = warp_img
    target_img_layer[:, :, i] = target
    rmax_shift = cmax_shift = 1
    while ~(rmax_shift <= threshold and cmax_shift <= threshold):
        target = (target + warp_img + refer) / 3
        warp_block_stack = create_image_block_stack(
            warp_img, block_row, block_col)
        warp_img, rmax_shift, cmax_shift, rtmap, ctmap = blocks_registration(
            warp_block_stack, target_block_stack, target, method, r, para.show_pyramid_fig
        )
        print("第{}层位移: (r = {}, c = {})".format(i + 1, rmax_shift, cmax_shift))
        count += 1
        rtmap += rtmap
        ctmap += ctmap
        if (count > para.pyramid_max_try) and (rtmap.any() >= 1 or ctmap.any() >= 1):
            layer_state[i] = 0
            warp_img = warp_img_layer[:, :, i]
            target = target_img_layer[:, :, i]
            print("第{}层位移有误，重新划分".format(i + 1))
            break
    block_row = block_row + 1
    block_col = block_col + 1

    target_block_stack = create_image_block_stack(target, block_row, block_col)
# %%
plt.figure(figsize=(10, 10))
plt.imshow(np.abs(first_warp_img - warp_img))
plt.axis("off")
plt.show()
target = refer
# %%
print(calculate_MI(target, first_warp_img))
print(calculate_MI(target, warp_img))
# %%
print(calculate_NCC(target, first_warp_img))
print(calculate_NCC(target, warp_img))
# %%
print(calculate_NMI(target, first_warp_img))
print(calculate_NMI(target, warp_img))
# %%
print(calculate_MSD(target, first_warp_img))
print(calculate_MSD(target, warp_img))
# %%
tup = [(i, layer_state[i]) for i in range(len(layer_state))]
k = np.max([j for j, n in tup if n == 1])
print("max layer: {}".format(k + 1))
for j, n in tup:
    if n == 1:
        print(
            "layer: {} , block size: {}".format(
                j + 1,
                (
                    int(np.ceil(target.shape[0] / (j + block_row_ini))),
                    int(np.ceil(target.shape[1] / (j + block_col_ini))),
                ),
            )
        )
# %%
tf.imwrite(para.save_img_path, warp_img.astype(np.uint8))
