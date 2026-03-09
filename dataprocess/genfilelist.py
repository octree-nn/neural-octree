import os
import random

def get_filtered_files(root_dir, exclude_file="pcd.pth"):
    """
    遍历目录及其所有子目录，获取文件名不为exclude_file的文件路径
    返回格式为：子目录/文件名
    """
    filtered_files = []
    
    # 使用os.walk递归遍历所有子目录
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 排除指定的文件名
            if file != exclude_file:
                # 获取相对于根目录的路径
                relative_path = os.path.relpath(os.path.join(root, file), root_dir)
                # 替换路径分隔符为/，确保跨平台一致性
                relative_path = relative_path.replace(os.sep, '/')
                filtered_files.append(relative_path)
    
    return filtered_files

def split_and_save_dataset(file_list, train_ratio=0.8, output_dir="."):
    """
    将文件列表按比例随机划分为训练集和测试集，并保存到文件
    """
    # 随机打乱文件列表以确保随机划分
    random.shuffle(file_list)
    
    # 计算训练集大小
    train_size = int(len(file_list) * train_ratio)
    
    # 划分训练集和测试集
    train_files = file_list[:train_size]
    test_files = file_list[train_size:]
    
    # 保存训练集到train.txt
    with open(os.path.join(output_dir, "train.txt"), "w", encoding="utf-8") as f:
        for file_path in train_files:
            f.write(file_path + "\n")
    
    # 保存测试集到test.txt
    with open(os.path.join(output_dir, "test.txt"), "w", encoding="utf-8") as f:
        for file_path in test_files:
            f.write(file_path + "\n")
    
    return train_files, test_files

def main():
    # 设置根目录路径（请修改为您的实际路径）
    root_directory = "/data/wangjh/NeuOctree_1208/cache/depth5/02958343_car"  # 替换为您的目录路径
    
    # 步骤1：获取所有符合条件的文件
    print("正在遍历目录并过滤文件...")
    all_files = get_filtered_files(root_directory, "pcd.pth")
    print(f"找到 {len(all_files)} 个符合条件的文件")
    
    # 步骤2：随机划分并保存数据集
    print("正在随机划分数据集...")
    train_set, test_set = split_and_save_dataset(all_files, train_ratio=0.8)
    
    # 步骤3：输出结果统计
    print("\n数据集划分完成！")
    print(f"训练集数量: {len(train_set)}")
    print(f"测试集数量: {len(test_set)}")
    print(f"划分比例: {len(train_set)}:{len(test_set)} (约4:1)")
    print(f"文件已保存到: train.txt 和 test.txt")

if __name__ == "__main__":
    main()