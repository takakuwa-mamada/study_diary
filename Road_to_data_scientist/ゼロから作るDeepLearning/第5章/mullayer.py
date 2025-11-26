import numpy as np
import matplotlib.pyplot as plt

class MulLayer:
    def __init__(self):
        self.x = None 
        self.y = None
        
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        
        return dx, dy
    
class AddLayer:
    def __init__(self):
        pass
    
    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# レイヤの生成
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
mul_tax_layer = MulLayer()  
add_apple_orange_layer = AddLayer()

# 順伝播
apple_price = mul_apple_layer.forward(apple, apple_num) 
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

# 逆伝播
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)

print(f"合計金額: {price}")
print(f"りんごの個数の勾配: {dapple_num}")  
print(f"りんごの単価の勾配: {dapple}")
print(f"オレンジの単価の勾配: {dorange}")
print(f"オレンジの個数の勾配: {dorange_num}")
print(f"消費税の勾配: {dtax}")

