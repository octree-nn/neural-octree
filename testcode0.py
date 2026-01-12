import torch

def test_normal_correction():
    """测试法向量校正功能"""
    # 案例1：单个点，法向量方向错误（应被翻转）
    pin = torch.tensor([[0.0, 0.0, 0.0]])  # 表面点
    pout = torch.tensor([[0.0, 0.0, 5.0]])  # 观察点（在z轴正方向）
    normals = torch.tensor([[0.0, 0.0, 1.0]])  # 法向量指向z轴正方向（错误，指向观察者）
    
    # 保存原始法向量用于比较
    normals_original = normals.clone()
    
    # 执行校正代码
    view_dirs = (pout - pin) / torch.norm(pout - pin, dim=-1, keepdim=True)
    normal_dot_view = (normals * view_dirs).sum(dim=-1)
    wrong_normal_mask = (normal_dot_view > 0)
    normals[wrong_normal_mask] = -normals[wrong_normal_mask]
    # print(normals)
    
    # 验证结果
    assert wrong_normal_mask.item() == True, "掩码应标记错误法向量"
    assert torch.all(normals == torch.tensor([[0.0, 0.0, -1.0]])), "法向量应被翻转为负z方向"
    print("✅ 案例1通过：错误法向量已校正")
    
    # 案例2：法向量方向正确（不应被翻转）
    normals_correct = torch.tensor([[0.0, 0.0, -1.0]])  # 法向量背离观察者
    normal_dot_view_correct = (normals_correct * view_dirs).sum(dim=-1, keepdim=True)
    wrong_normal_mask_correct = (normal_dot_view_correct > 0)
    
    assert wrong_normal_mask_correct.item() == False, "正确法向量不应被标记"
    print("✅ 案例2通过：正确法向量未改变")
    
    # 案例3：批量数据处理（多个点）
    pin_batch = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pout_batch = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    normals_batch = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])  # 第一个错误，第二个正确
    
    view_dirs_batch = (pout_batch - pin_batch) / torch.norm(pout_batch - pin_batch, dim=-1, keepdim=True)
    normal_dot_view_batch = (normals_batch * view_dirs_batch).sum(dim=-1)
    wrong_normal_mask_batch = (normal_dot_view_batch > 0)
    normals_batch[wrong_normal_mask_batch] = -normals_batch[wrong_normal_mask_batch]
    print(normals_batch)
    
    expected_batch = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])  # 两个法向量均应背离观察者
    assert torch.all(normals_batch == expected_batch), "批量处理结果不符预期"
    print("✅ 案例3通过：批量数据处理正确")
    
    # 案例4：边界条件（点积为零，法向量与视角垂直）
    normals_edge = torch.tensor([[1.0, 0.0, 0.0]])  # 法向量与视角方向垂直
    normal_dot_view_edge = (normals_edge * view_dirs).sum(dim=-1, keepdim=True)
    wrong_normal_mask_edge = (normal_dot_view_edge > 0)
    
    assert wrong_normal_mask_edge.item() == False, "点积为零时不应翻转"
    print("✅ 案例4通过：边界条件处理正确")

if __name__ == "__main__":
    test_normal_correction()
    print("所有测试案例通过！代码功能正常。")