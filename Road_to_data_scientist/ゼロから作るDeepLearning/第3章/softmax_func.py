import numpy as np
import matplotlib.pyplot as plt

a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)  # 指数関数を適用
sum_exp_a = np.sum(exp_a)

y = exp_a / sum_exp_a  # 各要素を指数関数の和で割る
print(y)  # ソフトマックス関数の出力

def softmax(a):
    C = np.max(a)
    exp_a = np.exp(a - C)  # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
# 出力が0から1の範囲に収まり，全ての出力の和が1になることを確認．これは確率を表せることができるということ