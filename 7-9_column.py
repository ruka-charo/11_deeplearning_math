import numpy as np


#%%
'''ベクトル間の内積'''
#w = (1, 2)
w = np.array([1, 2])
print('ベクトルw：', w)
print('ベクトルwの次元数：', w.shape)

#x = (3, 4)
x = np.array([3, 4])
print('ベクトルx：', x)
print('ベクトルxの次元数：', x.shape)

#y = w・x (内積)
y = w @ x
print(y)


#%%
'''行列とベクトルの内積'''
#Xは３行2列の行列
X = np.array([[1, 2], [3, 4], [5, 6]])
print('行列X\n', X)
print('行列の要素数：', X.shape)

#内積計算
Y = X @ w
print(Y)
print(Y.shape)


#%%
'''転置行列と内積'''
#転置行列の作成
XT = X . T
print('元の行列\n', X)
print('転置行列\n', XT)

yd = np.array([1, 2, 3])
print('ベクトル：', yd)

#勾配値の計算(一部)
grad = XT @ yd
print(grad)
