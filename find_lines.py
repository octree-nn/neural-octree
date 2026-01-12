import os

# 请用户修改以下路径为实际路径
train_file = '/data/wangjh/NeuOctree_1224/cache/depth5/02958343_car/filelist/train.txt'  # 输入文件路径（包含文件列表）
base_dir = '/home/wangjh/render_16views'  # 另一个目录路径（用于检查子目录是否存在）
output_file = 'found_lines.txt'  # 输出文件路径（保存匹配的行）

# 读取train.txt并处理每一行
found_lines = []  # 保存匹配的行
with open(train_file, 'r') as f:
    for line in f:
        line = line.strip()  # 去除行首尾空白字符（如换行符）
        if not line:  # 跳过空行
            continue
        
        # 提取子目录名：按'/'分割路径，取第一部分
        parts = line.split('/')
        if len(parts) < 2:  # 确保路径格式正确（至少包含一个'/'）
            print(f"跳过无效行：{line}")
            continue
        subdir = parts[0]  # 子目录名（如"7951309aad14b9f0afe717997470b28d"）
        
        # 构建子目录的完整路径（在base_dir下检查）
        target_path = os.path.join(base_dir, subdir)
        
        # 检查目标路径是否存在且是一个目录
        if os.path.exists(target_path) and os.path.isdir(target_path):
            found_lines.append(line)  # 记录匹配的行
            print(f"找到匹配目录：{subdir}")
        else:
            print(f"未找到目录：{subdir}")

# 将匹配的行写入新文件
with open(output_file, 'w') as out_f:
    for line in found_lines:
        out_f.write(line + '\n')

print(f"处理完成！共找到 {len(found_lines)} 个匹配项，结果已保存到 {output_file}")