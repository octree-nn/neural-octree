import numpy as np
from PIL import Image

def depth_to_mask(depth_image, threshold=0.99):
    """
    将深度图转换为二值掩码图。
    
    参数:
        depth_image (PIL.Image or np.ndarray): 输入的深度图像，单通道灰度图。
        threshold (float): 深度值小于该阈值的像素将被视为有效（前景）。
    """
    # 如果输入是PIL图像，转换为NumPy数组
    if isinstance(depth_image, Image.Image):
        depth_array = np.array(depth_image).astype(np.float32)
    else:
        depth_array = depth_image.astype(np.float32)
    
    # 归一化深度值到[0, 1]范围（假设深度图为16位或8位）
    if depth_array.max() > 255:
        depth_array = depth_array / 65535.0  # 16位图像
    else:
        depth_array = depth_array / 255.0     # 8位图像
    
    # 创建掩码：深度值小于阈值的像素设为1（前景），否则设为0（背景）
    mask_array = np.where(depth_array < threshold, 1, 0).astype(np.uint8) * 255  # 转换为0-255范围
    
    # 转换回PIL图像
    mask_image = Image.fromarray(mask_array, mode='L')

    
    return mask_image

# 示例用法
if __name__ == "__main__":
    # 读取深度图像
    depth_img_path = "depth_image.png"  # 替换为实际深度图路径
    depth_image = Image.open(depth_img_path).convert('L')  # 转为单通道灰度图
    
    # 转换为掩码图
    mask_image = depth_to_mask(depth_image, threshold=0.99)
    
    # 保存掩码图像
    mask_img_path = "mask_image.png"  # 替换为保存掩码图的路径
    mask_image.save(mask_img_path)
    
    print(f"掩码图已保存到 {mask_img_path}")