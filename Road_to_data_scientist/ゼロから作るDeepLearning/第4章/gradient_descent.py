import numpy as np 
import matplotlib.pyplot as plt

def numerical_gradient(f, x): # 数値微分, f: 微分したい関数, x: 微分する点
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # xと同じ形状の配列を生成
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x + h)
        
        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x - h)
        
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val # 値を元に戻す
    
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100): # 勾配降下法, f: 最適化したい関数, init_x: 初期値, lr: 学習率, step_num: 繰り返し回数
    x = init_x
    container = []
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        container.append(x.copy())
    return x, container

def function_2(x): # 最適化したい関数
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0]) # 初期値
result, container = gradient_descent(function_2, init_x, lr=0.1, step_num=100)
print("最適化されたx:", result)
# print("container:", container)
# y = np.arange(-5, 5.5, 0.5)

# plt.plot([-5, 5], [0, 0], '--b') # x軸
# plt.plot([0, 0], [-5, 5], '--b') # y軸
plt.plot(container)
plt.show()