import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.array([1.0, 0.5])  # 入力値
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # 1層目の重み
B1 = np.array([0.1, 0.2, 0.3])  # 1層目のバイアス

A1 = np.dot(X, W1) + B1  # 行列の内積を計算し，バイアスを足す
Z1 = sigmoid(A1)  # 活性化関数を通す

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])  # 2層目の重み
B2 = np.array([0.1, 0.2])  # 2層目のバイアス

A2 = np.dot(Z1, W2) + B2  # 行列の内積を計算し，バイアスを足す
Z2 = sigmoid(A2)  # 活性化関数を通す

def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])  # 3層目の重み
B3 = np.array([0.1, 0.2])  # 3層目のバイアス

A3 = np.dot(Z2, W3) + B3  # 行列の内積を計算し，バイアスを足す
Y = identity_function(A3)  # 恒等関数を通す

print(Y)  # 最終的な出力を表示