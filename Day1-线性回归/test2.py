import matplotlib.pyplot as plt

# 任务：实现线性回归
# y = w0 + w1 * x1 + w2 * x2

X = [[10,3],[20,3],[25,3],[28,2.5],[30,2],[35,2.5],[40,2.5]]
y = [60,85,100,120,140,145,163]
w = [0.0,0.0,0.0]
lr = 0.0012 # 学习率
num_iterations = 10000 # 迭代次数

# 0.001和10000拟合出来的效果较好
# 大于0.001 梯度更新过大
# 大于10000 过拟合

# 梯度下降
for i in range(num_iterations):
    # 预测值
    y_pred = [w[0]+w[1]*x[0]+w[2]*x[1] for x in X]
    # 计算损失
    loss = sum((y_pred[j]-y[j]) ** 2 for j in range(len(y))) / len(y)
    # 计算梯度
    grad_w0 = 2 * sum(y_pred[j]-y[j] for j in range(len(y))) / len(y)
    grad_w1 = 2 * sum((y_pred[j]-y[j]) * X[j][0] for j in range(len(y))) / len(y)
    grad_w2 = 2 * sum((y_pred[j]-y[j]) * X[j][1] for j in range(len(y))) / len(y)
    # 更新参数
    w[0] -= lr * grad_w0
    w[1] -= lr * grad_w1
    w[2] -= lr * grad_w2
    # 打印参数
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss}")
# 最终参数
print(f"Final parameters: w0 = {w[0]}, w1 = {w[1]}, w2 = {w[2]}")


# 打印最终拟合函数
X_1 = [x[0] for x in X]
X_2 = [x[1] for x in X]

plt.scatter(X_1, y, color='blue', label='Data Points')

y_fit = [w[0] + w[1] * x1 + w[2] *x2 for x1, x2 in zip(X_1, X_2)]
plt.plot(X_1, y_fit, color='red',label='Fitted Curve')

plt.xlabel('X1')
plt.ylabel('y')
plt.title('Data and Fitted Curve')
plt.legend()

plt.show()