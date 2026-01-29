import cv2
import numpy as np
import argparse

# 参数设置
parser = argparse.ArgumentParser(description='Concatenate GT and Predicted Images')
parser.add_argument('--object', type=str, default='Armadillo', help='Object name')
parser.add_argument('--config', type=str, default='d6_lr0005_0118', help='Configuration name')
parser.add_argument('--idx', type=int, default=0, help='View index')
args = parser.parse_args()

def read_tgt_image(object_name: str, config_name: str, idx: int):
    # img = cv2.imread(f'./training_views/{config_name}/{object_name}/{idx}.png')
    imgdir = f'./logs/{config_name}/rendering_views/{object_name}/{idx:03d}.png'
    img = cv2.imread(imgdir)
    norm = img[512:, 512:1024]
    not_hit_mask = img[512:, 1024:].astype(np.int32).sum(axis=-1) < 127 * 3
    # print(not_hit_mask.sum())
    norm[not_hit_mask] = 0
    norm = norm.astype(np.float64)
    # print(norm.shape)
    # print(norm.shape, norm.min(), norm.max(), norm.mean(), norm.std())
    return norm

def read_pred_image(object_name: str, config_name: str, idx: int):
    # img = cv2.imread(f'./training_views/{config_name}/{object_name}/{idx}.png')
    imgdir = f'./logs/{config_name}/rendering_views/{object_name}/{idx:03d}.png'
    img = cv2.imread(imgdir)
    norm = img[:512, 512:1024]
    not_hit_mask = img[:512, 1024:].astype(np.int32).sum(axis=-1) < 127 * 3
    # print(not_hit_mask.sum())
    norm[not_hit_mask] = 0
    norm = norm.astype(np.float64)
    # print(norm.shape)
    # print(norm.shape, norm.min(), norm.max(), norm.mean(), norm.std())
    return norm


def calculate_psnr(img1, img2):

    # 计算均方误差（MSE）
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return float('inf')  # 完全相同，PSNR 为无穷大

    # 计算 PSNR
    PIXEL_MAX = 255.0
    psnr = 10 * np.log10((PIXEL_MAX ** 2) / mse)
    return psnr

# 示例用法
if __name__ == "__main__":
    from tqdm import tqdm

    # config_name = 'L_depth5_fdim12_M1'
    # config_name = 'L_depth5_fdim32_VR11030100'
    config_name = args.config
    object_name = args.object
    view_idx = args.idx
    tgt_norm = read_tgt_image(object_name, config_name, view_idx)
    pred_norm = read_pred_image(object_name, config_name, view_idx)
    psnr_norm = calculate_psnr(tgt_norm, pred_norm)
    print(f"PSNR of normal for {object_name} view {view_idx}: {psnr_norm:.2f} dB")
    # config_name = 'L_depth5_fdim32_VR111611000-ginssim0p5norm'
    # config_name = 'L_depth5_fdim16_LPE202511091930'

    # objects = [
    #     # 'Armadillo',
    #     # 'armadillo-gt',
    #     args.object,
    #     # 'Buddha',
    #     # 'Dragon',
    #     # 'Einstein',
    #     # 'Lucy',
    #     # 'Metatron',
    #     # 'XYZ_dragon'
    # ]
    
    # for object_name in objects:
    #     print(object_name)
    #     try:
    #         total_psnr = 0.0
    #         for view_idx in tqdm(range(0, 256)):
    #             tgt_norm = read_tgt_image(object_name, config_name, view_idx)
    #             pred_norm = read_pred_image(object_name, config_name, view_idx)
    #             total_psnr += calculate_psnr(tgt_norm, pred_norm)
    #         psnr_norm = total_psnr / 256
    #         print(psnr_norm)
    #     except Exception as e:
    #         print(e)

    # in_path = 'depth2_512_d'
    # print(f"Calculating PSNR of {in_path}")
    # psnr_color = calculate_psnr(f"./sum_img/{in_path}/sum_color_6.png")
    # print(f"PSNR of color: {psnr_color:.2f} dB")
    # psnr_norm = calculate_psnr(f"./sum_img/{in_path}/sum_norm_6.png")
    # print(f"PSNR of normal: {psnr_norm:.2f} dB")
