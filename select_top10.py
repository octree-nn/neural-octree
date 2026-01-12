import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

def split_image(image):
    """
    将2048x1024的图像分割成8个512x512的块
    返回格式: 上面一行4个[渲染结果]，下面一行4个[Ground Truth]
    """
    height, width = image.shape[:2]
    if height != 1024 or width != 2048:
        raise ValueError(f"图像尺寸应为2048x1024，当前为{width}x{height}")
    
    blocks = []
    # 分割上面一行（渲染结果）
    for col in range(4):
        x_start = col * 512
        x_end = (col + 1) * 512
        top_block = image[0:512, x_start:x_end]
        blocks.append(top_block)
    
    # 分割下面一行（Ground Truth）
    for col in range(4):
        x_start = col * 512
        x_end = (col + 1) * 512
        bottom_block = image[512:1024, x_start:x_end]
        blocks.append(bottom_block)
    
    return blocks

def calculate_psnr(img1, img2):
    """
    计算两幅图像之间的PSNR值
    """
    # 确保图像数据类型一致
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # 计算均方误差
    mse = np.mean((img1 - img2) ** 2)
    
    # 处理MSE为0的情况
    if mse == 0:
        return float('inf')
    
    # 计算PSNR
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def process_directory(main_dir):
    """
    处理主目录下的所有子目录，计算PSNR并返回结果
    """
    main_path = Path(main_dir)
    results = []
    
    # 检查主目录是否存在
    if not main_path.exists():
        print(f"错误: 目录 {main_dir} 不存在")
        return results
    
    # 获取所有子目录
    subdirs = [d for d in main_path.iterdir() if d.is_dir()]
    print(f"找到 {len(subdirs)} 个子目录")
    
    for subdir in subdirs:
        image_path = subdir / "0.png"
        
        # 检查图像文件是否存在
        if not image_path.exists():
            print(f"警告: {image_path} 不存在，跳过")
            continue
        
        try:
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"警告: 无法读取 {image_path}，跳过")
                continue
            
            # 分割图像
            blocks = split_image(image)
            
            # 提取关注的区域（第二列和第三列）
            # 上面一行：索引1和2（第二列和第三列）- 渲染结果
            top_col2 = blocks[1]  # 第二列
            top_col3 = blocks[2]  # 第三列
            
            # 下面一行：索引5和6（第二列和第三列）- Ground Truth
            bottom_col2 = blocks[5]  # 第二列
            bottom_col3 = blocks[6]  # 第三列
            
            # 计算两对图像的PSNR
            psnr_col2 = calculate_psnr(top_col2, bottom_col2)
            psnr_col3 = calculate_psnr(top_col3, bottom_col3)
            
            # 计算平均PSNR
            avg_psnr = (psnr_col2 + psnr_col3) / 2
            
            results.append({
                'subdir': subdir.name,
                'path': str(subdir),
                'psnr_col2': psnr_col2,
                'psnr_col3': psnr_col3,
                'avg_psnr': avg_psnr
            })
            
            print(f"处理完成: {subdir.name}, 平均PSNR: {avg_psnr:.2f} dB")
            
        except Exception as e:
            print(f"处理 {subdir.name} 时出错: {str(e)}")
            continue
    
    return results

def main():
    main_dir = "./logs/depth9-5-proj/rendering_views"
    
    # 处理所有子目录
    results = process_directory(main_dir)
    
    if not results:
        print("未找到有效结果")
        return
    
    # 按平均PSNR降序排序
    results.sort(key=lambda x: x['avg_psnr'], reverse=True)
    
    # 输出前10名结果
    print("\n" + "="*80)
    print("平均PSNR最高的十个子目录:")
    print("="*80)
    print(f"{'排名':<4} {'子目录名':<20} {'PSNR第二列(dB)':<15} {'PSNR第三列(dB)':<15} {'平均PSNR(dB)':<15}")
    print("-"*80)
    
    for i, result in enumerate(results[:10], 1):
        print(f"{i:<4} {result['subdir']:<20} {result['psnr_col2']:<15.2f} {result['psnr_col3']:<15.2f} {result['avg_psnr']:<15.2f}")
    
    # 保存详细结果到文件
    output_file = "psnr_results.csv"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("子目录,路径,PSNR第二列,PSNR第三列,平均PSNR\n")
        for result in results:
            f.write(f"{result['subdir']},{result['path']},{result['psnr_col2']:.4f},{result['psnr_col3']:.4f},{result['avg_psnr']:.4f}\n")
    
    print(f"\n详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main()