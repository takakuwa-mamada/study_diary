import numpy as np
def AND(x1, x2): # ANDゲート
    w1, w2, theta = 0.5, 0.5, 0.7 
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

print(f"ANDゲートの結果：{AND(1, 1)}")

def OR(x1, x2): # ORゲート
    w1, w2, theta = 0.5, 0.5, 0.4
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
    
print(f"ORゲートの結果：{OR(0, 1)}")

# 重みとバイアスの導入
def AND2(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7 # バイアス -発火の度合いを調整する値
    tmp = np.sum(w*x) + b
    print(f"tmpの値：{tmp}")
    if tmp <= 0:
        return 0
    else:
        return 1

print(f"AND2ゲートの結果：{AND2(0, 1)}")

def NAND(x1, x2): # NANDゲート
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
#XORはパーセプトロンでは表現できない．しかし，層を増やすことで表現可能になる．
#要するに，単層のパーセプトロンでは非線形領域は分離できないが，多層にすることで非線形領域も分離できるようになる．

def XOR(x1, x2): # XORゲート
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y