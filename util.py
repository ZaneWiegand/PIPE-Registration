# %%
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.transform import warp
import numpy as np
# %%


def create_image_block_stack(img, row, col):
    imgsize_row, imgsize_col = img.shape[0], img.shape[1]
    blocksize_row = int(np.ceil(imgsize_row / row))
    blocksize_col = int(np.ceil(imgsize_col / col))
    pad_img = np.zeros([blocksize_row * row, blocksize_col * col])
    pad_img[0:imgsize_row, 0:imgsize_col] = img
    stack_img = np.zeros([row, col, blocksize_row, blocksize_col])
    for r in range(row):
        for c in range(col):
            stack_img[r, c, :, :] = pad_img[
                r * blocksize_row: (r + 1) * blocksize_row,
                c * blocksize_col: (c + 1) * blocksize_col,
            ]

    return stack_img


def stitch_block(stack_img, origin):
    # stack_img是四维向量，前两维储存位置，后两维储存图像块
    # origin用于获取原图尺寸
    num_row, num_col, block_row, block_col = stack_img.shape
    full_img = np.zeros([num_row * block_row, num_col * block_col])
    for i in range(num_row):
        for j in range(num_col):
            full_img[
                i * block_row: (i + 1) * block_row, j * block_col: (j + 1) * block_col
            ] = stack_img[i, j, :, :]
    full_img = full_img[0: origin.shape[0], 0: origin.shape[1]]
    return full_img


def fft_CMM(f, g):
    # 根据 Fast Normalized Cross-Correlation J. P. Lewis Industrial Light & Magic
    fmean = np.mean(f)
    gmean = np.mean(g)
    f = f - fmean
    g = g - gmean
    F = np.fft.fft2(f)
    G = np.fft.fft2(g)
    xx = F * np.conj(G)
    CMMmap = np.fft.fftshift(np.fft.ifft2(xx))
    return np.abs(CMMmap)


def PPMCC(obj, ref, r, c):
    row, col = ref.shape
    if r >= 0 and c >= 0:
        obj_sub = obj[0: row - r, 0: col - c]
        ref_sub = ref[r:, c:]
    elif r >= 0 and c < 0:
        c = -c
        obj_sub = obj[0: row - r, c:]
        ref_sub = ref[r:, 0: col - c]
    elif r < 0 and c >= 0:
        r = -r
        obj_sub = obj[r:, 0: col - c]
        ref_sub = ref[0: row - r, c:]
    else:
        r = -r
        c = -c
        obj_sub = obj[r:, c:]
        ref_sub = ref[0: row - r, 0: col - c]
    return np.sum(
        (obj_sub - np.mean(obj_sub)) * (ref_sub - np.mean(ref_sub))
    ) / (np.sqrt(
        np.sum((obj_sub - np.mean(obj_sub)) ** 2)
        * np.sum((ref_sub - np.mean(ref_sub)) ** 2)
    )+1e-8)


def n_CMM(CMM, PPMCC_max, PPMCC_min):
    if np.max(CMM) == np.min(CMM):
        nCMM = CMM / (np.max(CMM)+1e-8)
    else:
        nCMM = (CMM - np.min(CMM)) / (np.max(CMM) - np.min(CMM))
    nCMM = nCMM * (PPMCC_max - PPMCC_min) + PPMCC_min
    return nCMM


def blocks_registration(obj_stack, ref_stack, target, method, r, flag=False):
    num_row, num_col, nr, nc = ref_stack.shape
    CMM_stack = np.zeros_like(ref_stack)
    nCMM_stack = np.zeros_like(ref_stack)
    r_shift_stack = np.zeros([num_row, num_col, 1, 1])
    c_shift_stack = np.zeros([num_row, num_col, 1, 1])
    for i in range(num_row):
        for j in range(num_col):
            obj_copy = obj_stack[i, j, :, :]
            ref_copy = ref_stack[i, j, :, :]
            CMM_stack[i, j, :, :] = fft_CMM(obj_copy, ref_copy)
            maxloc = divmod(np.argmax(CMM_stack[i, j, :, :]), nc)
            r_max = int(np.round(maxloc[0] - nr / 2))
            c_max = int(np.round(maxloc[1] - nc / 2))
            minloc = divmod(np.argmin(CMM_stack[i, j, :, :]), nr)
            r_min = int(np.round(minloc[0] - nr / 2))
            c_min = int(np.round(minloc[1] - nc / 2))
            PPMCC_max = PPMCC(
                obj_copy,
                ref_copy,
                r_max,
                c_max,
            )
            PPMCC_min = PPMCC(
                obj_copy,
                ref_copy,
                r_min,
                c_min,
            )
            nCMM_stack[i, j, :, :] = n_CMM(
                CMM_stack[i, j, :, :], PPMCC_max, PPMCC_min)

            nCMM_stack_single = nCMM_stack[i, j, :, :]

            z = nCMM_stack_single[nr//2-r:nr//2+r+1, nc//2-r:nc//2+r+1]
            y = np.linspace(-r, r, 2*r+1)
            x = np.linspace(-r, r, 2*r+1)
            x, y = np.meshgrid(x, y)
            B = np.zeros([5, 5])
            B[0, 0] = np.sum(x**4)
            B[0, 1] = B[1, 0] = np.sum(x**2*y**2)
            B[0, 2] = B[2, 0] = np.sum(x**3)
            B[0, 3] = B[3, 0] = np.sum(x**2*y)
            B[0, 4] = B[4, 0] = np.sum(x**2)
            B[1, 1] = np.sum(y**4)
            B[1, 2] = B[2, 1] = np.sum(x*y**2)
            B[1, 3] = B[3, 1] = np.sum(y**3)
            B[1, 4] = B[4, 1] = np.sum(y**2)
            B[2, 2] = np.sum(x**2)
            B[2, 3] = B[3, 2] = np.sum(x*y)
            B[2, 4] = B[4, 2] = np.sum(x)
            B[3, 3] = np.sum(y**2)
            B[3, 4] = B[4, 3] = np.sum(y)
            B[4, 4] = np.sum(np.ones(z.shape))

            C = np.zeros([5, 1])
            C[0, 0] = np.sum(x**2*np.log(z+1e-8))
            C[1, 0] = np.sum(y**2*np.log(z+1e-8))
            C[2, 0] = np.sum(x*np.log(z+1e-8))
            C[3, 0] = np.sum(y*np.log(z+1e-8))
            C[4, 0] = np.sum(np.log(z+1e-8))

            K = np.dot(np.linalg.inv(B), C)

            if np.isnan(K).any():
                r_shift_stack[i, j, :, :] = 0
                c_shift_stack[i, j, :, :] = 0
            else:
                c_shift_stack[i, j, 0, 0] = K[2]/(-2*K[0])
                r_shift_stack[i, j, 0, 0] = K[3]/(-2*K[1])

    r_translation_map = stitch_block(
        r_shift_stack, np.zeros([num_row, num_col]))
    c_translation_map = stitch_block(
        c_shift_stack, np.zeros([num_row, num_col]))

    if flag:
        plt.figure(figsize=(8, 6))
        step = 1
        y, x = np.mgrid[
            : r_translation_map.shape[0]: step, : r_translation_map.shape[1]: step
        ]
        r_ = r_translation_map[::step, ::step]
        c_ = c_translation_map[::step, ::step]
        plt.quiver(
            x, y, -c_, -r_, color="r", units="dots", angles="xy", scale_units="xy"
        )
        # plt.title("Block Vector Field")
        plt.axis("off")
        plt.show()

    target_shape = (target.shape[1], target.shape[0])
    r_interpolated_translation_map = cv.resize(
        r_translation_map, target_shape, interpolation=method
    )
    c_interpolated_translation_map = cv.resize(
        c_translation_map, target_shape, interpolation=method
    )

    origin_img = stitch_block(obj_stack, target)
    row_coords, col_coords = np.meshgrid(
        np.arange(target.shape[0]), np.arange(target.shape[1]), indexing="ij"
    )
    warp_img = warp(
        origin_img,
        np.array(
            [
                row_coords + r_interpolated_translation_map,
                col_coords + c_interpolated_translation_map,
            ]
        ),
        mode="edge",
    )
    nCMM_full_map = stitch_block(
        nCMM_stack, np.zeros([num_row * nr, num_col * nc]))
    if flag:
        nvec = int(np.min(target.shape) / 30)
        step = max(target.shape[0] // nvec, target.shape[1] // nvec)
        y, x = np.mgrid[: target.shape[0]: step, : target.shape[1]: step]
        r_ = r_interpolated_translation_map[::step, ::step]
        c_ = c_interpolated_translation_map[::step, ::step]
        plt.figure(figsize=(12, 9))
        plt.quiver(
            x, y, -c_, -r_, color="r", units="dots", angles="xy", scale_units="xy"
        )
        # plt.title("Interpolated Vector Field")
        plt.axis("off")
        plt.show()
        plt.figure(figsize=(12, 9))
        plt.imshow(nCMM_full_map, vmin=0, vmax=1, cmap="gray")
        # plt.title("Stitched nCMM")
        plt.axis("off")
        plt.show()

    rmax_shift = np.max(np.abs(r_interpolated_translation_map))
    cmax_shift = np.max(np.abs(c_interpolated_translation_map))
    return (
        warp_img,
        rmax_shift,
        cmax_shift,
        r_interpolated_translation_map,
        c_interpolated_translation_map,
    )


def global_registration(obj, ref, method, flag=False):
    # TODO: SIFT
    if method == "SIFT":
        global_form = cv.SIFT_create()
    elif method == "ORB":
        global_form = cv.ORB_create()
    elif method == "KAZE":
        global_form = cv.KAZE_create()
    elif method == "AKAZE":
        global_form = cv.AKAZE_create()
    elif method == "BRISK":
        global_form = cv.BRISK_create()
    elif method == "nCCM":
        obj_block_stack = create_image_block_stack(obj, 1, 1)
        ref_block_stack = create_image_block_stack(ref, 1, 1)
        warp_img, _, _, _, _ = blocks_registration(
            obj_block_stack, ref_block_stack, ref, cv.INTER_LINEAR, flag
        )
        return warp_img
    else:
        return "ERROR"
    kp1, des1 = global_form.detectAndCompute(obj, None)
    kp2, des2 = global_form.detectAndCompute(ref, None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
    if flag:
        img = cv.drawMatchesKnn(
            obj,
            kp1,
            ref,
            kp2,
            good_matches,
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        img_obj = cv.drawKeypoints(
            obj, kp1, obj, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        )
        img_ref = cv.drawKeypoints(
            ref, kp1, ref, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        )
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(img_obj)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_ref)
        plt.axis("off")
        plt.tight_layout()
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    good_matches = np.squeeze(good_matches)
    # 物体特征点坐标
    ref_matched_kpts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    # 场景特征点坐标
    sensed_matched_kpts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    # 方法采用RANSAC计算投影矩阵，阈值设为5.0，即误差的2范数超过5.0，视为局外点
    H, status = cv.findHomography(
        ref_matched_kpts, sensed_matched_kpts, cv.RANSAC, 5.0)
    warp_img = cv.warpPerspective(
        obj, H, (ref.shape[1], ref.shape[0]
                 ), borderMode=cv.BORDER_REPLICATE
    )
    # TODO: SIFT end
    return warp_img


def calculate_MSD(target, warp):
    warp = warp.astype(np.uint8)
    target = target.astype(np.uint8)
    diff_pic = warp - target
    diff = np.mean(diff_pic * diff_pic)
    return diff


def calculate_MI(target, warp):
    target = target.astype(np.uint8)
    warp = warp.astype(np.uint8)
    target = np.reshape(target, -1)
    warp = np.reshape(warp, -1)
    size = warp.shape[-1]
    px = np.histogram(warp, 256, (0, 255))[0] / size
    py = np.histogram(target, 256, (0, 255))[0] / size
    hx = -np.sum(px * np.log(px + 1e-8))
    hy = -np.sum(py * np.log(py + 1e-8))

    hxy = np.histogram2d(warp, target, 256, [[0, 255], [0, 255]])[0]
    hxy /= 1.0 * size
    hxy = -np.sum(hxy * np.log(hxy + 1e-8))

    MI = hx + hy - hxy
    return MI


def calculate_NMI(target, warp):
    target = target.astype(np.uint8)
    warp = warp.astype(np.uint8)
    target = np.reshape(target, -1)
    warp = np.reshape(warp, -1)
    size = warp.shape[-1]
    px = np.histogram(warp, 256, (0, 255))[0] / size
    py = np.histogram(target, 256, (0, 255))[0] / size
    hx = -np.sum(px * np.log(px + 1e-8))
    hy = -np.sum(py * np.log(py + 1e-8))

    hxy = np.histogram2d(warp, target, 256, [[0, 255], [0, 255]])[0]
    hxy /= 1.0 * size
    hxy = -np.sum(hxy * np.log(hxy + 1e-8))

    NMI = 2*(hx+hy-hxy)/(hx + hy)
    return NMI


def calculate_NCC(target, warp):
    warp = warp.astype(np.uint8)
    target = target.astype(np.uint8)
    up = np.sum((target - np.mean(target)) * (warp - np.mean(warp)))
    down1 = np.sqrt(np.sum((target - np.mean(target))
                    * (target - np.mean(target))))
    down2 = np.sqrt(np.sum((warp - np.mean(warp)) * (warp - np.mean(warp))))
    return up / (down1 * down2)
