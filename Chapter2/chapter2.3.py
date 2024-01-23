# 2.3. 线性代数
# 2.3.1. 标量

import torch

# x = torch.tensor(3.0)
# y = torch.tensor(2.0)

# print(x + y)
# print(x - y)
# print(x * y)
# print(x / y)
# print(x ** y)


# 2.3.2. 向量
# x = torch.arange(4)
# print(x)
# print(x[3])


# 2.3.2.1. 长度、维度和形状
# x = torch.arange(4)
# print(len(x))
# print(x.shape)


# 2.3.3. 矩阵
# 大写字母X、Y来表示矩阵
# 矩阵 m * n，每行不会缺少列
# A = torch.arange(20).reshape(5, 4)
# print(A)
# print(A.T)

# B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
# print(B)
# print(B.T)
# print(B == B.T)

# 2.3.4. 张量
# X = torch.arange(24).reshape(2, 3, 4)
# print(X)
# print(X.shape)


# 2.3.5. 张量算法的基本性质
# A = torch.arange(20, dtype=torch.float32).reshape(5,4)
# # B = A.clone()
# print(A)
# print(A + A)
# print(A * A)


# X = torch.arange(24).reshape(2, 3, 4)
# print(X + 2)
# print(X * 2)
# print(X.shape)

# 2.3.6. 降维
# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# print(A)
# print(A.sum())
# print(A.shape)

# print(" ")
# A_sum_axis0 = A.sum(axis=0)
# print(A_sum_axis0)
# print(A_sum_axis0.shape)

# A_sum_axis1 = A.sum(axis=1)
# print(A_sum_axis1)
# print(A_sum_axis1.shape)


# print(" ")
# print(A.sum(axis=[0, 1]))

# print(" ")
# print(A.mean(), A.numel(), A.sum() / A.numel())

# print(" ")
# print(A.mean(axis = 0))
# print(A.sum(axis = 0) / A.shape[0])

# 2.3.6.1. 非降维求和
# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# sum_A = A.sum(axis=1, keepdims=True)
# print(A)
# print(sum_A)
# print(A / sum_A)
# print(A.cumsum(axis=0))


# 2.3.7. 点积（Dot Product）
# x = torch.arange(4, dtype=torch.float32)
# y = torch.ones(4, dtype=torch.float32)
# print(x, y, torch.dot(x, y))
# print(torch.sum(x * y))


# 2.3.8. 矩阵-向量积
# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# x = torch.arange(4, dtype=torch.float32)
# print(A, "\n", A.shape, "\n", x, "\n", x.shape, "\n", torch.mv(A, x))
# print(A[1])


# 2.3.9. 矩阵-矩阵乘法
# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# B = torch.ones(4, 3)
# result= torch.mm(A, B)
# print(A)
# print(B)
# print(result)


# 2.3.10. 范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))
print(torch.abs(u).sum())


print(torch.norm(torch.ones((4, 9))))