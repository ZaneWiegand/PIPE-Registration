# %%
import cv2 as cv
import tifffile as tf
import numpy as np
from util import global_registration, create_image_block_stack, blocks_registration, calculate_MI, calculate_MSD, calculate_NCC, calculate_NMI
import warnings
warnings.filterwarnings("ignore")
print("OK!")
# %%


class args(object):

    # 待配准图像路径
    float_img_path = 'float_img.tif'
    # 参考图像路径
    refer_img_path = 'refer_img.tif'
    # 保存图像路径
    save_img_path = 'save_img.tif'
    # 选择图片的类型，8位或者16位
    image_type = np.uint16  # np.uint8/np.uint16

    # 选择粗配准的方法，优先级SIFT>KAZE>AKAZE>BRISK>ORB>nCCM
    # method = "SIFT","KAZE","AKAZE","BRISK","ORB","nCCM"
    global_registration_method = 'SIFT'

    # 细配准最小位移阈值，默认采用0.25
    pyramid_threshold = 0.25  # (0,1)
    # 细配准分层数量，默认划分15层
    pyramid_layer = 15  # (1,inf)
    # 高斯拟合计算范围，用于生成位移矢量，默认为5
    pyramid_gaussian_range = 5
    # 位移矢量的插值方法，默认采用双三次插值
    pyramid_interp_method = cv.INTER_CUBIC
    # 一层最大尝试次数，默认为10
    pyramid_max_try = 10  # (1,inf)

    # 仅显示粗配准的中间结果，不影响配准最终结果，默认为False
    show_global_fig = False
    # 仅显示细配准的中间结果，不影响配准最终结果，默认为False
    show_pyramid_fig = False


para = args()
# %%
#!############################################################################################
#!############################################################################################
#!主程序开始
new = tf.imread(para.float_img_path).astype(para.image_type)
refer = tf.imread(para.refer_img_path).astype(para.image_type)
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
        if (count >= para.pyramid_max_try) and (rtmap.any() >= 1 or ctmap.any() >= 1):
            layer_state[i] = 0
            warp_img = warp_img_layer[:, :, i]
            target = target_img_layer[:, :, i]
            print("第{}层位移有误，重新划分".format(i + 1))
            break
    block_row = block_row + 1
    block_col = block_col + 1

    target_block_stack = create_image_block_stack(target, block_row, block_col)
# %%
# # warp_img
print('MI: {}'.format(calculate_MI(refer, warp_img)))
print('NCC: {}'.format(calculate_NCC(refer, warp_img)))
print('NMI: {}'.format(calculate_NMI(refer, warp_img)))
print('MSD: {}'.format(calculate_MSD(refer, warp_img)))
# # first_warp_img
# print('MI: {}'.format(calculate_MI(refer, first_warp_img)))
# print('NCC: {}'.format(calculate_NCC(refer, first_warp_img)))
# print('NMI: {}'.format(calculate_NMI(refer, first_warp_img)))
# print('MSD: {}'.format(calculate_MSD(refer, first_warp_img)))
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
tf.imwrite(para.save_img_path, warp_img.astype(para.image_type))
