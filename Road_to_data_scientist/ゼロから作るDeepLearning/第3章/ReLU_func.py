import numpy as np

def ReLU(x): #ReLU関数の実装
    return np.maximum(0, x) #0とxを比較し，大きい方を返す．この場合，xが0以下なら0，それ以外ならxを返す．

print(ReLU(-3))