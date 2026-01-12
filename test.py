import torch

# 假设您的神经网络预测出的 tensor a
a = torch.randn(5, requires_grad=True) # 假设 a 需要梯度
print("a:", a)

# 假设您已有的 tensor b
b = torch.ones(10, requires_grad=False) # 假设 b 本身也需要梯度
print("原始 b:", b)

# 定义需要替换的索引
indexes = torch.tensor([1, 3, 5, 7, 9])

# --- 方法1: 使用 torch.scatter 或 torch.index_copy ---
# 创建一个 b 的副本（或使用 b 本身），然后进行可导的索引操作
# c = b.clone() # 先克隆 b，避免原地修改原 b
b[indexes] = a # 现在这个操作是在新张量 c 上进行的，但问题依旧存在，不推荐直接使用。
# 更好的替代方法是 torch.index_copy 或构造掩码张量

# # --- 方法2: 使用 torch.where (更通用) ---
# # 创建一个与 b 同形状的全零张量，然后在指定索引处填入 a 的值
# mask = torch.zeros_like(b, dtype=torch.bool)
# mask[indexes] = True
# # 使用 torch.where：如果 mask 为 True，则取 a 的值；否则取 b 的值
# c = torch.where(mask, a, b) # 关键：c 的创建依赖于 a 和 b，计算图是连续的

# print("新张量 c:", c)

# 后续对 c 进行操作
loss = b.sum()
loss.backward()

# 检查梯度是否能传回 a
print("a.grad 是否存在:", a.grad is not None) # 应该为 True
print("a.grad:", a.grad)