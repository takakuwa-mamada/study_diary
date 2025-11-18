import numpy as np
import matplotlib.pyplot as plt
def step_function(x): #このやり方だと，np.arrayが対応できない
    
    if x > 0:
        return 1
    else:
        return 0

def new_step_function(x): #np.arrayに対応できる
    y = x > 0
    return y.astype(int)  # np.intは非推奨 → intを使用

x = np.arange(-5.0, 5.0, 0.1)
y = new_step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) #y軸の範囲を指定
plt.show()
