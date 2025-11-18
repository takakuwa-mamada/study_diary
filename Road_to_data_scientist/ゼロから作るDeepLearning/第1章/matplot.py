#Matplotlibを使った簡単なグラフの描画
import numpy as np
import matplotlib.pyplot as plt

#データの作成
x = np.arange(0, 6, 0.1) # 0から6まで0.1刻みの配列を作成
y = np.sin(x) # xの各要素に対してsin関数を適用

#グラフの作成
# plt.plot(x, y)
# plt.show() # グラフを表示

y1 = np.sin(x)
y2 = np.cos(x)

#複数のグラフを同じ座標軸に描画
plt.plot(x, y1, label='sin(x)') # sin関数のグラフ
plt.plot(x, y2, linestyle='--', label='cos(x)') # cos関数のグラフ
plt.xlabel('x axis') # x軸のラベル
plt.ylabel('y axis') # y軸のラベル
plt.title('Sine and Cosine Functions') # グラフのタイトル
plt.legend() # 凡例の表示
plt.grid() # グリッドの表示
plt.show() # グラフを表示