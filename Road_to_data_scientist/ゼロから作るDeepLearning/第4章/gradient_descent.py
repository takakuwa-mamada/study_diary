import numpy as np 
import matplotlib.pyplot as plt

# 日本語フォントの設定（文字化け対策）
plt.rcParams['font.family'] = 'MS Gothic'  # Windowsの場合
# または以下のいずれか（環境によって選択）
# plt.rcParams['font.family'] = 'Yu Gothic'  # 游ゴシック
# plt.rcParams['font.family'] = 'Meiryo'     # メイリオ
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策

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
    x = init_x.copy()  # 初期値をコピー（元の値を保持）
    container = [x.copy()]  # 初期位置を記録
    
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

# containerをNumPy配列に変換（扱いやすくするため）
container = np.array(container)
print(f"更新回数: {len(container)-1}回")  # 初期位置を含むので-1
print(f"初期位置: ({init_x[0]:.4f}, {init_x[1]:.4f})")  # 元の初期値を表示
print(f"最終位置: ({result[0]:.4f}, {result[1]:.4f})")  # 最適化後の値を表示
print(f"初期関数値: {function_2(init_x):.4f}")
print(f"最終関数値: {function_2(result):.10f}")

# ========================================
# 可視化1: xの各成分の推移（2つのグラフ）
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# x[0]の推移
axes[0].plot(container[:, 0], marker='o', markersize=3)
axes[0].axhline(y=0, color='r', linestyle='--', label='目標値 (x₀=0)')
axes[0].set_xlabel('更新回数（イテレーション）')
axes[0].set_ylabel('x₀の値')
axes[0].set_title('x₀の推移')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# x[1]の推移
axes[1].plot(container[:, 1], marker='o', markersize=3, color='orange')
axes[1].axhline(y=0, color='r', linestyle='--', label='目標値 (x₁=0)')
axes[1].set_xlabel('更新回数（イテレーション）')
axes[1].set_ylabel('x₁の値')
axes[1].set_title('x₁の推移')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()

# ========================================
# 可視化2: 2次元空間での更新経路（等高線図付き）
# ========================================
plt.figure(figsize=(8, 8))

# 等高線を描画するためのグリッド作成
x0 = np.arange(-4, 4, 0.1)
x1 = np.arange(-4, 6, 0.1)
X0, X1 = np.meshgrid(x0, x1)

# 関数値を計算
Z = X0**2 + X1**2

# 等高線を描画
contour = plt.contour(X0, X1, Z, levels=20, colors='gray', alpha=0.5)
plt.clabel(contour, inline=True, fontsize=8)

# 勾配降下法の経路を描画
plt.plot(container[:, 0], container[:, 1], 'o-', color='red', 
         markersize=5, linewidth=2, label='更新経路')

# 開始点と終了点を強調
plt.plot(container[0, 0], container[0, 1], 'go', markersize=15, 
         label=f'開始点 ({container[0, 0]:.2f}, {container[0, 1]:.2f})')
plt.plot(container[-1, 0], container[-1, 1], 'r*', markersize=20, 
         label=f'終了点 ({container[-1, 0]:.2e}, {container[-1, 1]:.2e})')

# 最小値（目標）を表示
plt.plot(0, 0, 'bs', markersize=15, label='最小値 (0, 0)')

plt.xlabel('x₀', fontsize=12)
plt.ylabel('x₁', fontsize=12)
plt.title(f'勾配降下法の更新経路\n（学習率={0.1}, 反復回数={len(container)}）', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

# ========================================
# 可視化3: 関数値（損失）の推移
# ========================================
plt.figure(figsize=(10, 5))

# 各ステップでの関数値を計算
losses = [function_2(x) for x in container]

plt.plot(losses, marker='o', markersize=3, linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', label='最小値 (f=0)')
plt.xlabel('更新回数（イテレーション）', fontsize=12)
plt.ylabel('関数値 f(x)', fontsize=12)
plt.title('勾配降下法による関数値の減少', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.yscale('log')  # 対数スケール（変化を見やすくするため）
plt.tight_layout()
plt.show()