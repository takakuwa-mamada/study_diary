"""
gradient_simple.py
==================
ニューラルネットワークにおける勾配計算の基礎を学ぶプログラム

このプログラムの目的：
1. 簡単なニューラルネットワーク（2入力→3出力）を作る
2. 損失関数を計算する
3. 重みに対する勾配（∂損失/∂重み）を数値微分で求める
4. 勾配が「重みをどう調整すべきか」を教えてくれることを理解する
"""

import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt

# 日本語フォントの設定（文字化け対策）
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ========================================
# ユーティリティ関数
# ========================================

def softmax(x):
    """
    ソフトマックス関数：出力を確率に変換
    
    例：[1.0, 2.0, 3.0] → [0.09, 0.24, 0.67]
        合計が1になる（確率として解釈可能）
    """
    c = np.max(x)
    exp_x = np.exp(x - c)  # オーバーフロー対策
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

def cross_entropy_error(y, t):
    """
    交差エントロピー誤差：予測と正解の「ズレ」を測る
    
    Parameters:
    -----------
    y : 予測確率（例：[0.1, 0.3, 0.6]）
    t : 正解ラベル（one-hot、例：[0, 0, 1]）
    
    Returns:
    --------
    誤差：値が小さいほど予測が正確
    """
    delta = 1e-7  # log(0)対策（log(0)は計算できないので微小値を足す）
    return -np.sum(t * np.log(y + delta))

def numerical_gradient(f, x):
    """
    数値微分による勾配計算（多次元配列対応版）
    
    Parameters:
    -----------
    f : 関数（損失関数など）
    x : 変数（重み行列など）
    
    Returns:
    --------
    grad : xと同じ形状の勾配配列
    
    動作の仕組み：
    ------------
    各要素について、微小変化（h=0.0001）を与えて
    関数値がどれだけ変化するかを計算（中心差分）
    
    例：W[0,0]を0.0001増やしたとき、損失が0.002増えた
        → grad[0,0] = 0.002 / 0.0001 = 20
        → W[0,0]を減らせば損失が減る（勾配が正なので）
    """
    h = 1e-4  # 0.0001（微小な変化量）
    grad = np.zeros_like(x)  # xと同じ形状のゼロ配列

    # np.nditer: 多次元配列を要素ごとに処理するイテレータ
    # multi_index: (0,0), (0,1), ... のようなインデックスを取得
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index  # 現在の要素のインデックス
        tmp_val = x[idx]      # 元の値を保存

        # f(x+h)を計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)を計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        # 中心差分: [f(x+h) - f(x-h)] / 2h
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        
        x[idx] = tmp_val  # 値を元に戻す（重要！）
        it.iternext()     # 次の要素へ

    return grad

# ========================================
# 簡単なニューラルネットワーククラス
# ========================================

class simpleNet:
    """
    2入力・3出力の単層ニューラルネットワーク
    
    ネットワーク構造：
    -----------------
    入力層(2)  →  出力層(3)
      x[0] ──┐
             ├──→ [y[0], y[1], y[2]]
      x[1] ──┘
    
    重み行列W（2×3）：
      W = [[w00, w01, w02],   ← x[0]から各出力への重み
           [w10, w11, w12]]   ← x[1]から各出力への重み
    """
    
    def __init__(self):
        """重みをランダム初期化"""
        self.W = np.random.randn(2, 3)  # 標準正規分布からサンプリング

    def predict(self, x):
        """
        順伝播：入力xから出力を計算
        
        計算式：y = x・W（内積）
        
        例：x=[0.6, 0.9], W=[[1,2,3],[4,5,6]] の場合
            y = [0.6×1+0.9×4, 0.6×2+0.9×5, 0.6×3+0.9×6]
              = [4.2, 5.7, 7.2]
        """
        return np.dot(x, self.W)

    def loss(self, x, t):
        """
        損失関数の計算
        
        ステップ1：順伝播で予測値を計算
        ステップ2：ソフトマックスで確率に変換
        ステップ3：交差エントロピー誤差を計算
        
        Parameters:
        -----------
        x : 入力データ（例：[0.6, 0.9]）
        t : 正解ラベル（one-hot、例：[0, 0, 1]）
        
        Returns:
        --------
        loss : 損失（誤差）の値
        """
        z = self.predict(x)     # 予測値
        y = softmax(z)          # 確率に変換
        loss = cross_entropy_error(y, t)  # 誤差計算
        return loss

# ========================================
# 実行例：勾配の計算と確認
# ========================================

# ニューラルネットワークの作成
net = simpleNet()
print("初期の重み行列W:")
print(net.W)
print()

# サンプル入力データ（2次元ベクトル）
x = np.array([0.6, 0.9])

# 順伝播：予測値の計算
p = net.predict(x)
print("予測値（各クラスのスコア）:", p)

# 正解ラベル（one-hotベクトル）
# [0, 0, 1] → クラス2が正解
t = np.array([0, 0, 1])

# 損失関数の値を確認
loss = net.loss(x, t)
print("損失関数の値:", loss)
print()

# ========================================
# 勾配の計算（2つの方法）
# ========================================

# 方法1：通常の関数定義
def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print("重みWの勾配（方法1）:")
print(dW)
print()

# 方法2：lambda式（より簡潔）
f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print("重みWの勾配（方法2）:")
print(dW)
print()

# ========================================
# 勾配の解釈
# ========================================
# dW[i,j]の意味：
# - 正の値：W[i,j]を増やすと損失が増える → W[i,j]を減らすべき
# - 負の値：W[i,j]を増やすと損失が減る → W[i,j]を増やすべき
# - 絶対値が大きい：その重みが損失に大きく影響する
#
# 例：dW[0,2] = -5.2 の場合
#     → W[0,2]を1増やすと、損失が約5.2減る
#     → 重みを増やす方向に更新すべき
#
# 学習のステップ：
#     W_new = W_old - learning_rate × dW
#     （勾配の逆方向に重みを更新することで損失を減らす）