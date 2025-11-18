import numpy as np

x = np.array([1, 2, 3])
# print(type(x)) # <class 'numpy.ndarray'>

y = np.array([4, 5, 6])
#算術計算（ブロードキャスト）
# print(x + y) # [5 7 9]
# print(x * y) # [ 4 10 18]
# print(x / y) # [0.25 0.4  0.5 ]
# print(x - y) # [-3 -3 -3]

#注意点：配列の要素数は同じでなければならない

#NumpyのN次元配列
A = np.array([[1, 2, 3], [4, 5, 6]])
# print(A) #[[1 2 3] [4 5 6]]
# print(A.shape) #(2, 3)

B = np.array([[4, 5, 6], [7, 8, 9]])
# print(B) #[[4 5 6] [7 8 9]]
# print(B.shape) #(2, 3)

# print(A + B) # [[ 5  7  9] [11 13 15]]
# print(A * B) # [[ 4 10 18] [28 40 54]]

#行列の算術計算も，同じ形状の行列同士であれば，要素ごとに計算が行われる．

#-- ブロードキャスト ---
A = np.array([[1,2], [3,4]])
B = np.array([10, 20])

# print(A + B) # [[11 12] [13 24]]
# print(A * B) # [[10 40] [30 80]]

#要素のアクセス
# print(A[0]) # [1 2]
# print(A[0][1]) # 2

for row in A:
    # print(row)
    pass

AA = A.flatten() #
# print(AA) # [1 2 3 4]
# print(AA[np.array([0, 2, 3])]) # インデックスが0,2,3の要素を取得
# print(AA > 1) # 要素が1より大きいかどうかの真偽値を取得