# ニューラルネットワーク

## 第3章の概要

この章では、**ニューラルネットワーク**の仕組みを理解し、実装します。パーセプトロンでは重みを手動で設定していましたが、ニューラルネットワークは**データから重みを自動的に学習**できます。

第3章で学ぶこと：
- **活性化関数**の役割と種類
- **多層ニューラルネットワーク**の実装
- **MNISTデータセット**を使った手書き数字認識

---

## 1. パーセプトロンからニューラルネットワークへ

### パーセプトロンの限界

第2章で学んだパーセプトロンの問題点：
- 重みとバイアスを**手動で設定**する必要がある
- 複雑な問題では、適切なパラメータを見つけることが困難
- XORのような非線形問題も多層化で解決できるが、パラメータ設定が難しい

### ニューラルネットワークの革新

**重みの自動調整（学習）が可能**
- 訓練データから最適なパラメータを自動的に見つける
- 人間が手動で調整する必要がない
- より複雑なパターン認識が可能

**鍵となるのは「活性化関数」の導入**

---

## 2. 活性化関数（Activation Function）

活性化関数は、入力信号の総和を出力信号に変換する関数です。ニューラルネットワークの表現力を決定する重要な要素です。

### ニューロンの計算プロセス

```
1. 加重和の計算：    a = b + w₁x₁ + w₂x₂ + ... + wₙxₙ
2. 活性化関数の適用： y = h(a)
```

- `a`：入力信号の加重和（weighted sum）
- `h()`：活性化関数（activation function）
- `y`：出力信号

---

## 3. ステップ関数（Step Function）

**パーセプトロンで使われていた活性化関数**

### 定義

```
h(x) = { 0  (x ≤ 0)
       { 1  (x > 0)
```

### 実装（step_func.py）

```python
import numpy as np
import matplotlib.pyplot as plt

# シンプル版（NumPy配列に非対応）
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

# NumPy配列対応版
def new_step_function(x):
    y = x > 0
    return y.astype(int)  # bool型をint型に変換

# グラフ描画
x = np.arange(-5.0, 5.0, 0.1)
y = new_step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
```

### 特徴

- **0または1の2値のみ出力**
- 閾値で出力が急激に変化
- **非連続**な関数
- **微分不可能**（0の点で）

**問題点：** 微分できないため、学習（重みの調整）に使いにくい

---

## 4. シグモイド関数（Sigmoid Function）

**ニューラルネットワークでよく使われる滑らかな活性化関数**

### 定義

```
h(x) = 1 / (1 + exp(-x))
```

### 実装（sigmoid_func.py）

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# グラフ描画
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
```

### 特徴

- **0から1の間の実数値を出力**
- 滑らかなS字カーブ（Sigmoid = S字型）
- **連続かつ全域で微分可能**
- 入力が大きいと1に近づき、小さいと0に近づく
- **非線形**関数

### NumPyのブロードキャスト

シグモイド関数の実装は、NumPyのブロードキャストにより、スカラーでも配列でも同じコードで動作します。

```python
sigmoid(1.0)           # スカラー → 0.731...
sigmoid(np.array([1.0, 2.0, 3.0]))  # 配列 → array([0.731..., 0.880..., 0.952...])
```

---

## 5. ReLU関数（Rectified Linear Unit）

**現代のディープラーニングで最も人気のある活性化関数**

### 定義

```
h(x) = max(0, x) = { 0  (x ≤ 0)
                   { x  (x > 0)
```

**直感的な理解：**
- 入力が負なら0を出力（オフ状態）
- 入力が正ならそのまま出力（オン状態）
- 「ランプ関数」とも呼ばれる

### 実装（ReLU_func.py）

```python
import numpy as np
import matplotlib.pyplot as plt

def ReLU(x):
    return np.maximum(0, x)  # 0とxを比較し、大きい方を返す

# 動作確認
print(ReLU(-3))  # 0
print(ReLU(3))   # 3
print(ReLU(0))   # 0

# 配列でも動作
x = np.array([-2, -1, 0, 1, 2])
print(ReLU(x))   # [0 0 0 1 2]

# グラフ描画
x = np.arange(-5.0, 5.0, 0.1)
y = ReLU(x)
plt.plot(x, y)
plt.ylim(-1, 5)
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.title('ReLU Function')
plt.grid()
plt.show()
```

**グラフの特徴：**
```
  y
  5│         ╱
  4│       ╱
  3│     ╱
  2│   ╱
  1│ ╱
  0├─────────── x
  -5  -3  -1  0  1  3  5

x < 0 では y = 0（フラット）
x > 0 では y = x（直線、傾き1）
```

### 特徴

- **入力が0以下なら0、正なら入力をそのまま出力**
- 計算が非常にシンプルで高速（比較だけ）
- **勾配消失問題**を緩和
- 深いネットワークでも学習が安定
- **スパース性**（多くの出力が0になる）

### なぜReLUが主流なのか？

**1. 計算効率**
```python
# ReLU: 単純な比較
y = max(0, x)

# シグモイド: 指数計算が必要
y = 1 / (1 + exp(-x))  # exp()は重い計算
```

**2. 勾配消失問題の解決**

シグモイド関数の問題：
```
x が大きい（例：10）または小さい（例：-10）とき、
勾配（傾き）がほぼ0になる → 学習が進まない

  σ'(x)
   ↑
0.25│   ╱‾‾‾╲      勾配のグラフ
    │  ╱     ╲
  0 │─╱───────╲─→ x
    -5    0    5

x = ±5 で勾配がほぼ0
```

ReLUの勾配：
```
ReLU'(x) = { 0  (x < 0)
           { 1  (x > 0)

正の領域では常に勾配 = 1
→ 勾配が消失しない！
```

**3. 生物学的妥当性**
- 実際の脳のニューロンも「閾値を超えたら発火」という動作
- ReLUはこの動作に近い

**4. スパース性（Sparsity）**
```python
x = np.array([-2, -1, 0, 1, 2, 3])
y = ReLU(x)
# y = [0, 0, 0, 1, 2, 3]
#      ↑  ↑  ↑  活性化していないニューロン（50%）

# 利点：
# - メモリ効率が良い
# - 計算量が減る
# - 過学習の抑制
```

**5. 深いネットワークでの安定性**
```
浅いネットワーク（2〜3層）：シグモイドでも問題ない
深いネットワーク（10層以上）：ReLUが圧倒的に優位

理由：勾配消失問題が層を重ねるごとに悪化
     ReLUなら勾配が安定
```

### ReLUの改良版

実際のディープラーニングでは、ReLUの改良版も使われます：

**Leaky ReLU：**
```python
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

# x < 0 でも小さな勾配（alpha）を持つ
# 「死んだReLU問題」を緩和
```

**ELU（Exponential Linear Unit）：**
```python
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# 負の領域で滑らか
# 平均出力が0に近い
```

**Swish（Google開発）：**
```python
def swish(x):
    return x * sigmoid(x)

# ReLUとシグモイドの良いとこ取り
# 深いネットワークで高性能
```

---

## 6. ステップ関数 vs シグモイド関数

### 共通点

1. **非線形関数である**
   - 線形関数では層を重ねる意味がない
   - 非線形性により複雑な表現が可能

2. **単調増加**
   - 入力が大きくなると出力も大きくなる

3. **出力範囲が限定的**
   - ステップ関数：{0, 1}
   - シグモイド関数：(0, 1)

### 相違点

| 特徴 | ステップ関数 | シグモイド関数 |
|------|-------------|---------------|
| **出力** | 0または1（2値） | 0〜1の実数 |
| **連続性** | 非連続 | 連続 |
| **微分可能性** | 不可 | 可能 |
| **滑らかさ** | 急激に変化 | 滑らか |
| **用途** | パーセプトロン | ニューラルネットワーク |

### なぜ非線形が重要か？

**線形関数を使うと、何層重ねても1層と同じになってしまいます。**

例：線形関数 `h(x) = cx` を3層重ねると：

```
y = h(h(h(x))) = c × c × c × x = c³x
```

これは `y = ax`（a = c³）という1層の線形関数と等価です。

**非線形関数により、層を重ねることで表現力が指数的に増大します。**

---

## 7. 多次元配列の計算

ニューラルネットワークの実装には、NumPyによる行列演算が不可欠です。

### 行列の内積（ドット積）

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 行列の積
C = np.dot(A, B)
print(C)  # [[19 22]
          #  [43 50]]
```

### 形状の確認と制約

```python
A = np.array([[1, 2, 3], [4, 5, 6]])  # 2×3行列
B = np.array([[1, 2], [3, 4], [5, 6]])  # 3×2行列

print(A.shape)  # (2, 3)
print(B.shape)  # (3, 2)

C = np.dot(A, B)
print(C.shape)  # (2, 2)
```

**内積のルール：** `(m, n) × (n, k) = (m, k)`
- 第1行列の列数 = 第2行列の行数 でなければならない

---

## 8. 3層ニューラルネットワークの実装

### ネットワーク構造

```
入力層（2ニューロン）
    ↓
第1層（隠れ層、3ニューロン）
    ↓
第2層（隠れ層、2ニューロン）
    ↓
第3層（出力層、2ニューロン）
```

### 実装（3layer_neural_network.py）

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

# 入力値
X = np.array([1.0, 0.5])

# 第1層
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X, W1) + B1  # 加重和
Z1 = sigmoid(A1)          # 活性化

# 第2層
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

# 第3層（出力層）
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)  # 恒等関数（回帰問題用）

print(Y)  # 最終出力
```

### 記号の意味

- `X`：入力
- `W`：重み行列（Weight）
- `B`：バイアス（Bias）
- `A`：活性化関数に入る前の加重和
- `Z`：活性化関数を通過した後の出力
- `Y`：最終出力

### 各層の形状

```python
X.shape   # (2,)
W1.shape  # (2, 3)
A1.shape  # (3,)
Z1.shape  # (3,)
W2.shape  # (3, 2)
A2.shape  # (2,)
Z2.shape  # (2,)
W3.shape  # (2, 2)
Y.shape   # (2,)
```

**形状の整合性が取れていることが重要です。**

---

## 9. 出力層の設計

出力層の活性化関数は、**解きたい問題の種類**によって変えます。

### 9.1 恒等関数（Identity Function）

**回帰問題（実数値の予測）に使用**

```python
def identity_function(x):
    return x
```

入力をそのまま出力します。

**用途：** 株価予測、気温予測、売上予測など

### 9.2 ソフトマックス関数（Softmax Function）

**多クラス分類問題に使用**

#### 定義

```
yₖ = exp(aₖ) / Σⁿᵢ₌₁ exp(aᵢ)
```

各クラスに属する確率を出力します。

#### 実装（softmax_func.py）

```python
import numpy as np

# 基本実装
a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)
sum_exp_a = np.sum(exp_a)
y = exp_a / sum_exp_a

print(y)  # [0.018... 0.245... 0.735...]
print(np.sum(y))  # 1.0

# オーバーフロー対策版
def softmax(a):
    C = np.max(a)
    exp_a = np.exp(a - C)  # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
```

#### オーバーフロー対策

指数関数は値が急激に大きくなるため、オーバーフロー（桁あふれ）が発生する可能性があります。

```python
# 問題：大きな数値の指数
np.exp(1000)  # inf（無限大）

# 解決策：最大値を引く
C = np.max(a)
np.exp(a - C)  # 安全
```

数学的に、ソフトマックス関数は定数Cを足しても引いても結果は同じです：

```
yₖ = exp(aₖ + C) / Σ exp(aᵢ + C) = exp(aₖ) / Σ exp(aᵢ)
```

#### ソフトマックス関数の特徴

1. **出力の合計が1**：確率として解釈可能
2. **各要素は0〜1の範囲**
3. **大きな入力値ほど大きな出力値**
4. **相対的な大小関係を保持**

**用途：** 手書き数字認識、画像分類、テキスト分類など

---

## 10. MNISTデータセット

**MNIST**は、手書き数字（0〜9）の画像データセットで、機械学習の「Hello World」とも呼ばれています。

### データセットの構成

- **訓練画像**：60,000枚
- **テスト画像**：10,000枚
- **画像サイズ**：28×28ピクセル（グレースケール）
- **ラベル**：0〜9の数字

### データの読み込み（mnist_train.py）

```python
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000,)
print(x_test.shape)   # (10000, 784)
print(t_test.shape)   # (10000,)
```

#### load_mnist関数の引数

- `normalize`：画像のピクセル値を0.0〜1.0に正規化するか
- `flatten`：画像を1次元配列に変換するか（28×28 → 784）
- `one_hot_label`：ラベルをone-hot表現にするか

### 画像の表示（mnist_show.py）

```python
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # ラベル（例：5）

img = img.reshape(28, 28)  # 784 → 28×28に変換
img_show(img)
```

**ポイント：**
- `flatten=True`で読み込んだ画像は1次元配列（784要素）
- 表示するには`reshape(28, 28)`で2次元に戻す必要がある

---

## 11. ニューラルネットワークの推論

事前学習済みの重みを使って、手書き数字を認識します。

### 実装（neuralnet_mnist.py）

```python
import numpy as np
import pickle
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    C = np.max(a)
    exp_a = np.exp(a - C)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False
    )
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

# テストデータで評価
x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  # 最も確率が高いクラスのインデックス
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
# 実行結果：Accuracy:0.9352
```

### 処理の流れ

1. **データ読み込み**
   - テストデータ（10,000枚）を取得
   - 正規化（0.0〜1.0）とフラット化（28×28 → 784）

2. **ネットワーク初期化**
   - 事前学習済みの重み（`sample_weight.pkl`）を読み込み

3. **予測（順伝播）**
   - 3層ニューラルネットワークで計算
   - 出力層でソフトマックス関数を適用

4. **精度計算**
   - 予測と正解ラベルを比較
   - 正解率を計算

### 結果

```
Accuracy:0.9352
```

**93.52%の精度**で手書き数字を認識できました！

### np.argmax関数

```python
y = np.array([0.1, 0.3, 0.2, 0.4])
np.argmax(y)  # 3（最大値のインデックス）
```

ソフトマックス関数の出力から、最も確率の高いクラスを選択します。

---

## 12. バッチ処理

現在の実装は1枚ずつ処理していますが、**バッチ処理**でまとめて処理すると効率的です。

### バッチ処理のメリット

1. **計算の高速化**
   - NumPyの内部最適化が効く
   - 並列計算が可能

2. **メモリの効率化**
   - データの読み込みを減らせる

### バッチ処理の実装例

```python
batch_size = 100

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
```

**`axis=1`の意味：** 各データ（行）ごとに最大値のインデックスを求める

---

## データサイエンスにおける重要ポイント

### 1. 活性化関数の選択

| 層 | 活性化関数 | 理由 |
|----|-----------|------|
| 隠れ層 | **ReLU** | 勾配消失問題の回避、高速 |
| 隠れ層 | シグモイド | 古典的、現在はあまり使われない |
| 出力層（回帰） | 恒等関数 | 実数値を出力 |
| 出力層（2値分類） | シグモイド | 0〜1の確率 |
| 出力層（多クラス分類） | **ソフトマックス** | 各クラスの確率 |

### 2. 正規化の重要性

```python
load_mnist(normalize=True)  # ピクセル値を0〜1に正規化
```

**理由：**
- 学習が安定する
- 勾配の爆発・消失を防ぐ
- 異なるスケールの特徴量を統一

### 3. 形状（shape）の管理

ニューラルネットワークのデバッグで最も重要なのは**形状の確認**です。

```python
print(f"Input: {x.shape}")
print(f"Weight: {W.shape}")
print(f"Output: {y.shape}")
```

**エラーの多くは形状の不一致から発生します。**

### 4. 推論 vs 学習

- **推論（Inference）**：学習済みモデルで予測（この章で実装）
- **学習（Training）**：データから重みを調整（第4章以降）

### 5. 前処理の重要性

機械学習では、**前処理が性能の80%を決める**と言われます：

- **正規化**：値の範囲を統一
- **標準化**：平均0、分散1に変換
- **次元削減**：不要な特徴量を除去
- **データ拡張**：訓練データを増やす

---

## まとめ

**第3章で学んだこと：**

1. **活性化関数の種類と特徴**
   - ステップ関数：パーセプトロン（非連続）
   - シグモイド関数：ニューラルネットワーク（滑らか）
   - ReLU関数：現代のディープラーニング（高速・効果的）

2. **非線形性の重要性**
   - 線形関数では層を重ねる意味がない
   - 非線形により複雑な表現が可能

3. **多層ニューラルネットワークの実装**
   - NumPyによる行列演算
   - 順伝播（forward propagation）
   - 形状管理の重要性

4. **出力層の設計**
   - 回帰：恒等関数
   - 2値分類：シグモイド関数
   - 多クラス分類：ソフトマックス関数

5. **MNISTによる実践**
   - データの読み込みと前処理
   - 事前学習済みモデルによる推論
   - 93.52%の認識精度を達成

6. **バッチ処理**
   - 複数データをまとめて処理
   - 計算効率の向上

**次章への準備：**

第4章では、**ニューラルネットワークの学習**について学びます。現在は事前学習済みの重み（`sample_weight.pkl`）を使用していますが、訓練データから最適な重みを自動的に見つける方法を習得します。

**学習のキーワード：**
- 損失関数（Loss Function）
- 勾配降下法（Gradient Descent）
- 誤差逆伝播法（Backpropagation）

---

## 補足：実装で使う主要なNumPy関数

```python
# 配列生成
np.array([1, 2, 3])
np.zeros((2, 3))
np.ones((2, 3))
np.arange(0, 10, 0.1)

# 行列演算
np.dot(A, B)          # 内積（行列の積）
A @ B                 # 内積（Python 3.5以降）
A * B                 # 要素ごとの積

# 数学関数
np.exp(x)             # 指数関数
np.maximum(0, x)      # 要素ごとの最大値（ReLU）
np.sum(x)             # 合計
np.sum(x, axis=0)     # 列ごとの合計
np.sum(x, axis=1)     # 行ごとの合計
np.max(x)             # 最大値
np.argmax(x)          # 最大値のインデックス
np.argmax(x, axis=1)  # 各行の最大値のインデックス

# 形状操作
x.shape               # 形状の取得
x.reshape(2, 3)       # 形状の変換
x.flatten()           # 1次元化

# 比較・論理演算
x > 0                 # 要素ごとの比較
x == y                # 要素ごとの等価性
```

**これらの関数を使いこなすことが、効率的なニューラルネットワーク実装の鍵です。**
