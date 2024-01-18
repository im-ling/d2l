import torch
import math
import numpy
# 2.1. 数据操作
# 2.1.1 入门

# x = torch.arange(12)
# print(x)
# print(x.shape)
# print(x.numel())
# print(x.reshape(3, 4))
# print(x.reshape(3, -1))
# print(x.reshape(-1, 4))

# print(torch.zeros([2,3,4]))

# print(torch.zeros([2,3,4,2]))

# print(torch.randn(3, 4))

# x = torch.tensor([[2, 1, 4, 3],
#                   [1, 2, 3, 4],
#                   [4, 3, 2, 1],])
# print(x)


# 2.1.2 运算符
# x = torch.tensor([1.0, 2, 4, 8])
# y = torch.tensor([2, 2, 2, 2])


# print(x + y)
# print(x - y)
# print(x * y)
# print(x / y)
# print(x ** y)

# # 自然对数e的x次方
# print(torch.exp(x))
# # 自然对数e为底
# print(math.log(2.7183))


# X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
# Y = torch.tensor([[2.0, 1, 4, 3],
#                   [1, 2, 3, 4],
#                   [4, 3, 2, 1],])
# print(torch.cat((X, Y), dim=0),)
# print(torch.cat((X, Y), dim=1),)

# print(X == Y)
# print(X.sum())


# 2.1.3. 广播机制

# X = torch.arange(3).reshape((3, 1))
# Y = torch.arange(2)
# print(X)
# print(Y)
# print(X + Y)

# 2.1.4. 索引和切片
# X = torch.arange(12).reshape((4,3))
# print(X)
# print(X[-1])
# print(X[1:3])
# print(X[1,2])
# X[1,2] = 99
# print(X)
# X[0:2, :] = 88
# print(X)

# # 2.1.5. 节省内存
# Y = torch.tensor([0, 2])
# X = torch.ones([2])
# before = id(Y)
# # id(Y) == before
# print(before)
# Y = Y + X
# print("\nY = Y + X")
# print(Y)
# print(id(Y))
# print("\nY += X")
# Y += X
# print(Y)
# print(id(Y))
# Y[:] = X + Y
# print("\nY[:] = X + Y")
# print(Y)
# print(id(Y))


# 2.1.6. 转换为其他Python对象
# array = [[1, 2, 3, 4], [5, 6, 7, 8]]
# X = torch.tensor(array)
# A = X.numpy()
# B = torch.tensor(A)
# print(type(A))
# print(type(B))
# print(A)
# print(B)

# print('\nX[0,0] = 99')
# X[0,0] = 99
# print(X)
# print(A)
# print(B)


# print('\nXA[0,1] = 99')
# A[0,1] = 99
# print(X)
# print(A)
# print(B)


# print('\nXA[0,1] = 99')
# B[0,2] = 99
# print(X)
# print(A)
# print(B)



# 反过来不能修改
A = numpy.array([1,2,3,4])
B = torch.tensor(A)
print(A)
print(B)

print("\nA[0] = 99")
A[0] = 99
print(A)
print(B)


print("\nB[1] = 88")
B[1] = 88
print(A)
print(B)
