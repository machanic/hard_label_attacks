import torch

# 假设图片 x 和掩码 mask
x = torch.randn(3, 224, 224)  # 3通道图片
mask = torch.randn(3, 224, 224)  # 3通道掩码矩阵

# 计算每个像素点的余弦相似度
# 分别计算分子和分母
numerator = (x * mask).sum(dim=0)  # 通道维度求和
x_norm = torch.sqrt((x ** 2).sum(dim=0))  # 图片的 L2 范数
print(x_norm.shape)
mask_norm = torch.sqrt((mask ** 2).sum(dim=0))  # 掩码的 L2 范数
denominator = x_norm * mask_norm

# 避免分母为0（防止数值问题）
denominator = torch.clamp(denominator, min=1e-8)

# 计算余弦相似度
cosine_similarity = numerator / denominator

print(cosine_similarity)  # 输出为 [224, 224]
