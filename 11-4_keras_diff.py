import numpy as np

# ネイピア数を底とする指数関数の定義
def f(x):
    return np.exp(x)

# 微少な数 hの定義
h = 0.001

# f'(0)の近似計算
# f'(0) = f(0) = 1に近い値になるはず
diff = (f(0 + h) - f(0 - h))/(2 * h)

# 結果の確認
print(diff)
