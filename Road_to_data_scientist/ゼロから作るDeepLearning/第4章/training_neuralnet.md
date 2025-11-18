# ニューラルネットワークの学習

## 第4章の概要

この章では、ニューラルネットワークが**データから学習する仕組み**を理解します。第3章では事前学習済みの重みを使いましたが、ここでは**訓練データから最適な重みを自動的に見つける方法**を学びます。

第4章で学ぶこと：
- **損失関数**：学習の指標
- **数値微分**と**勾配**：パラメータの更新方向
- **勾配降下法**：最適化アルゴリズム
- **学習アルゴリズムの実装**

---

## 1. 学習とは

### ニューラルネットワークの特徴

**ニューラルネットワークの最大の特徴は「学習できる」こと**

従来の機械学習：
```
生データ → 特徴量抽出（人間が設計） → 機械学習 → 出力
```

ディープラーニング：
```
生データ → ニューラルネットワーク → 出力
        （特徴量も自動で学習）
```

**特徴量（Feature）**：入力データから本質的なデータを的確に抽出する変換器

### 学習の目的

**訓練データから最適な重みパラメータを見つけること**

- 「最適」とは、損失関数の値が最小になること
- 損失関数を指標として、重みを調整していく

---

## 2. 訓練データとテストデータ

### データの分割

機械学習では、データを2つに分けます：

- **訓練データ（Training Data）**：学習に使用するデータ
- **テストデータ（Test Data）**：学習済みモデルの性能評価に使用

**重要：** テストデータは学習に使わない！

### なぜ分けるのか？

**汎化能力（Generalization Ability）を評価するため**

- **汎化能力**：未知のデータに対する認識能力
- 訓練データだけで評価すると、特定のデータにだけ最適化される危険性

### 過学習（Overfitting）

**過学習**：訓練データに過剰に適応し、汎化能力が低下すること

```
訓練データの精度：99%  ← 高い！
テストデータの精度：70%  ← 低い... 過学習！
```

**過学習の兆候：**
- 訓練データの精度は高いが、テストデータの精度が低い
- モデルが訓練データを「暗記」している状態

---

## 3. 損失関数（Loss Function）

### 損失関数とは

**損失関数**は、ニューラルネットワークの性能の「悪さ」を示す指標です。

- 損失関数の値が小さいほど、モデルの性能が良い
- 学習の目標：損失関数を最小化する重みパラメータを見つける

**損失関数が学習の鍵となる指標です。**

### なぜ認識精度を使わないのか？

**認識精度は微分できないため、学習に使えません。**

例：100個のデータで33個正解（精度33%）
- 重みをわずかに変更しても、精度は33%のまま（離散的）
- 損失関数なら連続的に変化するため、微分可能

**重み更新には連続的な値の変化が必要 → 損失関数を使う**

---

## 4. 二乗和誤差（Mean Squared Error）

### 定義

```
E = (1/2) Σₖ (yₖ - tₖ)²
```

- `yₖ`：ニューラルネットワークの出力
- `tₖ`：正解ラベル（教師データ）
- `k`：データの次元数

### 実装

```python
import numpy as np

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)
```

### 例

```python
# 正解ラベルは「2」
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# 例1：「2」の確率が最も高い
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
mean_squared_error(np.array(y1), np.array(t))  # 0.097...

# 例2：「7」の確率が最も高い
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
mean_squared_error(np.array(y2), np.array(t))  # 0.597...
```

**y1の方が損失が小さい → y1の方が正解に近い**

---

## 5. 交差エントロピー誤差（Cross Entropy Error）

### 定義

```
E = -Σₖ tₖ log(yₖ)
```

- `yₖ`：ニューラルネットワークの出力
- `tₖ`：正解ラベル（one-hot表現）
- `log`：自然対数（底がe）

### one-hot表現

正解ラベルを1つだけ1で、他を0で表現：

```python
# 「2」がクラスの場合
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
```

### 実装

```python
import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7  # log(0)を防ぐための微小値
    return -np.sum(t * np.log(y + delta))
```

### 例

```python
# 正解ラベルは「2」
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# 例1：「2」の確率が最も高い
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
cross_entropy_error(np.array(y1), np.array(t))  # 0.510...

# 例2：「7」の確率が最も高い
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
cross_entropy_error(np.array(y2), np.array(t))  # 2.302...
```

**y1の方が損失が小さい → y1の方が正解に近い**

### なぜ交差エントロピーを使うのか？

**分類問題では交差エントロピー誤差の方が適している**

- 確率分布の差異を測るのに適している
- 学習が安定しやすい
- ソフトマックス関数との相性が良い

---

## 6. ミニバッチ学習

### 問題：訓練データが大量

MNISTの例：
- 訓練データ：60,000枚
- 全データの損失関数を計算するのは時間がかかる

### 解決策：ミニバッチ学習

**一部のデータ（ミニバッチ）を使って損失関数を計算**

```
全データの損失 ≈ ミニバッチの損失 × データ数 / バッチサイズ
```

### 実装（mini_batch.py）

```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

train_size = x_train.shape[0]  # 60000
batch_size = 10

# ランダムに10個のインデックスを選択
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
```

**`np.random.choice(60000, 10)`**：0〜59999の中から10個をランダムに選ぶ

### バッチ対応の交差エントロピー誤差（cross_entropy.py）

```python
import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
```

**ポイント：**
- 1次元配列（単一データ）を2次元配列（バッチ）に変換
- バッチサイズで割って平均を取る

### one-hotではないラベルへの対応

```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    
    # tがone-hotの場合
    # return -np.sum(t * np.log(y + 1e-7)) / batch_size
    
    # tがラベル（0, 1, 2, ...）の場合
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```

**`y[np.arange(batch_size), t]`**：各データの正解ラベルに対応する出力を取得

---

## 7. なぜ損失関数を設定するのか？

### 認識精度ではダメな理由

**認識精度は微分がほとんど0になる**

- 認識精度：正解か不正解かの離散的な値（33%、34%、35%...）
- わずかな重みの変化では、精度は変わらないことが多い
- 勾配がほとんど0 → 重みをどう更新すべきかわからない

### 損失関数の利点

**損失関数は連続的に変化する**

- 重みをわずかに変えれば、損失関数も連続的に変化
- 微分が意味を持つ
- 勾配（損失を減らす方向）がわかる

**ニューラルネットワークの学習では、連続的な値の変化が不可欠です。**

---

## 8. 数値微分（Numerical Differentiation）

### 微分とは

**微分**：ある瞬間の変化量（傾き）

```
f'(x) = lim[h→0] (f(x+h) - f(x)) / h
```

### 数値微分の実装

```python
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)
```

**ポイント：**
- `h`は微小な値（10⁻⁴程度）
- 中心差分を使う：`(f(x+h) - f(x-h)) / 2h`
  - 前方差分：`(f(x+h) - f(x)) / h` より精度が高い

### 例：y = x² の微分

```python
def function_1(x):
    return x ** 2

# x=5での微分
numerical_diff(function_1, 5)  # 9.999999999999787 ≈ 10

# 解析的な微分：f'(x) = 2x → f'(5) = 10
```

**数値微分は真の微分の近似値です。**

---

## 9. 偏微分（Partial Derivative）

### 偏微分とは

**多変数関数において、1つの変数についての微分**

例：`f(x₀, x₁) = x₀² + x₁²`

- `x₀`についての偏微分：`∂f/∂x₀ = 2x₀`
- `x₁`についての偏微分：`∂f/∂x₁ = 2x₁`

### 実装

```python
def function_2(x):
    return x[0]**2 + x[1]**2

# x₀=3, x₁=4の点での偏微分
def function_tmp1(x0):
    return x0**2 + 4.0**2

numerical_diff(function_tmp1, 3.0)  # 6.00000000... ≈ 6

def function_tmp2(x1):
    return 3.0**2 + x1**2

numerical_diff(function_tmp2, 4.0)  # 7.99999999... ≈ 8
```

---

## 10. 勾配（Gradient）

### 勾配とは

**すべての変数の偏微分をまとめたベクトル**

```
∇f = (∂f/∂x₀, ∂f/∂x₁, ..., ∂f/∂xₙ)
```

### 実装

```python
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # xと同じ形状の配列
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 値を元に戻す
        
    return grad
```

### 勾配の意味

**勾配は「関数の値を最も減らす方向」を示します。**

例：`f(x₀, x₁) = x₀² + x₁²`

```python
# (3, 4)での勾配
numerical_gradient(function_2, np.array([3.0, 4.0]))
# array([6., 8.])

# (0, 2)での勾配
numerical_gradient(function_2, np.array([0.0, 2.0]))
# array([0., 4.])

# (3, 0)での勾配
numerical_gradient(function_2, np.array([3.0, 0.0]))
# array([6., 0.])
```

**勾配の方向に進むと関数の値が増加し、逆方向に進むと減少します。**

### 勾配法（Gradient Method）

勾配の情報を使って関数の最小値を探す手法：

- **勾配降下法（Gradient Descent）**：最小値を探す
- **勾配上昇法（Gradient Ascent）**：最大値を探す

---

## 11. 勾配降下法（Gradient Descent）

### アルゴリズム

```
x ← x - η∇f(x)
```

- `x`：更新する変数（重みやバイアス）
- `η`（イータ）：学習率（Learning Rate）
- `∇f(x)`：勾配

**学習率**：1回の学習でどれだけ更新するかを決めるパラメータ

### 実装

```python
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        
    return x
```

**パラメータ：**
- `f`：最適化したい関数
- `init_x`：初期値
- `lr`：学習率（Learning Rate）
- `step_num`：繰り返し回数

### 例

```python
# f(x₀, x₁) = x₀² + x₁²の最小値を探す
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x, lr=0.1, step_num=100)
# array([-6.11110793e-10,  8.14814391e-10]) ≈ [0, 0]
```

**最小値(0, 0)に到達しました！**

### 学習率の重要性

**学習率が大きすぎる場合：**

```python
gradient_descent(function_2, init_x, lr=10.0, step_num=100)
# array([-2.58983747e+13, -1.29524862e+12])  発散！
```

**学習率が小さすぎる場合：**

```python
gradient_descent(function_2, init_x, lr=1e-10, step_num=100)
# array([-2.99999994,  3.99999992])  ほとんど更新されない
```

**適切な学習率の設定がとても重要です。**

---

## 12. ニューラルネットワークに対する勾配

### 重みに対する損失関数の勾配

ニューラルネットワークの学習目標：

```
損失関数 L に対する重み W の勾配 ∂L/∂W
```

**この勾配は「重みをどう変えれば損失が減るか」を示します。**

### 簡単なニューラルネットワークの例

```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 2×3の重み行列
        
    def predict(self, x):
        return np.dot(x, self.W)
        
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss
```

### 勾配の計算

```python
net = simpleNet()
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

# 勾配の計算
def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)
# [[ 0.21924763  0.14356247 -0.36281009]
#  [ 0.32887144  0.2153437  -0.54421514]]
```

**この勾配が「重みをどう更新すべきか」を教えてくれます。**

### 勾配の解釈

例：`dW[0, 2] = -0.36281009`

- 負の値 → `W[0, 2]`を増やせば損失が減る
- 値の大きさ → 変化の影響度

**勾配に従って重みを更新することで、損失を減らせます。**

---

## 13. 学習アルゴリズムの実装

### 前提

ニューラルネットワークには、適応可能な重みとバイアスがあり、訓練データに適応するように調整することを**学習**と呼びます。

### ニューラルネットワークの学習手順

**ステップ1：ミニバッチ**
訓練データからランダムに一部のデータを選ぶ

**ステップ2：勾配の計算**
各重みパラメータに対する損失関数の勾配を求める

**ステップ3：パラメータの更新**
勾配方向に重みパラメータを更新

**ステップ4：繰り返し**
ステップ1〜3を繰り返す

**これが「確率的勾配降下法（SGD: Stochastic Gradient Descent）」です。**

### 2層ニューラルネットワークのクラス

```python
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)
        
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
```

### 学習の実装

```python
# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1エポックあたりの繰り返し数
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配の計算
    grad = network.numerical_gradient(x_batch, t_batch)
    
    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 学習経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 1エポックごとに精度を計算
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train acc, test acc | {train_acc}, {test_acc}")
```

**エポック（Epoch）**：すべての訓練データを1回学習し終えた単位

---

## データサイエンスにおける重要ポイント

### 1. 損失関数の選択

| 問題 | 損失関数 |
|------|---------|
| **回帰** | 二乗和誤差（MSE） |
| **分類** | 交差エントロピー誤差 |

### 2. ハイパーパラメータ

**ハイパーパラメータ**：学習前に設定するパラメータ

- **学習率（Learning Rate）**：更新の大きさ
- **バッチサイズ**：一度に処理するデータ数
- **エポック数**：訓練データを何回繰り返すか
- **隠れ層のニューロン数**

**これらは試行錯誤で調整する必要があります。**

### 3. 過学習の防止

- **訓練データとテストデータを分ける**
- **検証データ（Validation Data）**でハイパーパラメータを調整
- 正則化（Regularization）を使用（第6章）
- Dropout（第6章）

### 4. 学習の可視化

**学習曲線（Learning Curve）**を描くことで、学習の進行状況を把握：

```python
import matplotlib.pyplot as plt

plt.plot(train_loss_list)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
```

### 5. ミニバッチ学習の利点

- **計算速度の向上**：全データより高速
- **メモリ効率**：大規模データセットでも扱える
- **汎化性能の向上**：ノイズによる正則化効果

### 6. 数値微分 vs 誤差逆伝播法

**数値微分**：
- 実装が簡単
- 計算が遅い
- 勾配の確認に使用

**誤差逆伝播法（第5章）**：
- 計算が高速
- 実装が複雑
- 実際の学習で使用

---

## まとめ

**第4章で学んだこと：**

1. **学習とは**
   - データから最適な重みパラメータを見つけること
   - 特徴量も自動で学習できる

2. **訓練データとテストデータ**
   - 汎化能力の評価
   - 過学習の検出

3. **損失関数**
   - 二乗和誤差（回帰問題）
   - 交差エントロピー誤差（分類問題）
   - なぜ認識精度ではダメか

4. **ミニバッチ学習**
   - 一部のデータで効率的に学習
   - ランダムサンプリング

5. **数値微分と勾配**
   - 微分：瞬間の変化量
   - 偏微分：多変数関数の微分
   - 勾配：すべての変数の偏微分

6. **勾配降下法**
   - 勾配に従ってパラメータを更新
   - 学習率の重要性

7. **学習アルゴリズムの実装**
   - ミニバッチ → 勾配計算 → 更新 → 繰り返し
   - 確率的勾配降下法（SGD）

**次章への準備：**

第5章では、**誤差逆伝播法（Backpropagation）**を学びます。数値微分より遥かに高速に勾配を計算できる手法で、実用的なディープラーニングに不可欠です。

**キーワード：**
- 計算グラフ
- 連鎖律（Chain Rule）
- 逆伝播
- 高速な勾配計算

---

## 補足：よく使う用語

### 学習関連

- **訓練（Training）**：モデルを学習させること
- **推論（Inference）**：学習済みモデルで予測すること
- **エポック（Epoch）**：全訓練データを1回学習した単位
- **イテレーション（Iteration）**：パラメータ更新1回

### パラメータ

- **重み（Weight）**：学習で調整されるパラメータ
- **バイアス（Bias）**：学習で調整されるパラメータ
- **ハイパーパラメータ**：学習前に設定するパラメータ

### 最適化

- **勾配降下法（Gradient Descent）**：基本的な最適化手法
- **SGD（Stochastic Gradient Descent）**：確率的勾配降下法
- **学習率（Learning Rate）**：更新の大きさを決めるパラメータ

### 評価

- **損失（Loss）**：モデルの性能の悪さを示す指標
- **精度（Accuracy）**：正解率
- **汎化（Generalization）**：未知データへの対応能力
