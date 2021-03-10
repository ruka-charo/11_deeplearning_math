import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D

#PDF出力用
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')


#学習用データ準備
iris = load_iris()
x_org, y_org = iris.data, iris.target
print('元データ', x_org.shape, y_org.shape)

#データ絞り込み
#クラス0、1のみ
#項目sepal_lengthとsepal_widthのみ
x_data, y_data = iris.data[:100,:2], iris.target[:100]
print('対象データ', x_data.shape, y_data.shape)

#ダミー変数を追加
x_data = np.insert(x_data, 0, 1.0, axis=1)
print('ダミー変数追加後', x_data.shape)

#元データのサイズ
print(x_data.shape, y_data.shape)

#学習データ、検証データに分割（シャッフルの同時に実施）
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=70, test_size=30, random_state=123
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#学習データの散布図表示
x_t0 = x_train[y_train == 0]
x_t1 = x_train[y_train == 1]
plt.figure(figsize=(6,6))
plt.scatter(x_t0[:,1], x_t0[:,2], marker='x', c='b', label='0 (setosa)')
plt.scatter(x_t1[:,1], x_t1[:,2], marker='o', c='k', label='1 (versicolor)')
plt.xlabel('sepal_length', fontsize=14)
plt.ylabel('sepal_width', fontsize=14)
plt.xticks(size=16)
plt.yticks(size=16)
plt.legend(fontsize=16)
plt.show()

#学習用変数の設定
x = x_train
yt = y_train

#入力データxの表示(ダミーデータを含む)
print(x[:5])

#正解値ytの表示
print(yt[:5])


'''評価'''
#シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#予測値の計算
def pred(x, w):
    return sigmoid(x @ w)

#損失関数(交差エントロピー関数)
def cross_entropy(yt, yp):
    #交差エントロピーの計算(この段階ではベクトル)
    ce1 = -(yt * np.log(yp) + (1 - yt) * np.log(1 - yp))
    #交差エントロピーベクトルの平均値を計算
    return np.mean(ce1)

#予測結果の確率値から0or1を判断する関数
def classify(y):
    return np.where(y < 0.5, 0, 1)

#モデルの評価を行う関数
def evaluate(xt, yt, w):
    #予測値の計算
    yp = pred(xt, w)
    #損失関数の計算
    loss = cross_entropy(yt, yp)
    #予測値(確率値)を0または1に変換
    yp_b = classify(yp)
    #精度の算出
    score = accuracy_score(yt, yp_b)
    return loss, score


#初期化処理
#標本数
M = x.shape[0]

#入力次元数(ダミー変数を含む)
D = x.shape[1]

#繰り返し回数
iters = 10000

#学習率
alpha = 0.01

#初期値
w = np.ones(D)

#評価結果記録用(損失関数と精度)
history = np.zeros((0, 3))


#繰り返しループ
for k in range(iters):
    #予測値の計算(8.6.1) (8.6.2)
    yp = pred(x, w)
    #誤差の計算(8.6.4)
    yd = yp - yt
    #勾配降下法の実施(8.6.6)
    w = w - alpha * (x.T @ yd) / M
    #ログ記録用
    if k % 10 == 0:
        loss, score = evaluate(x_test, y_test, w)
        history = np.vstack((history, np.array([k, loss, score])))
        print('iter = %d loss = %f score = %f' % (k, loss, score))


#損失関数値と精度の確認
print('初期状態: 損失関数:%f 精度:%f' % (history[0,1], history[0,2]))
print('最終状態: 損失関数:%f 精度:%f' % (history[-1,1], history[-1,2]))

#検証データを散布図用に準備
x_t0 = x_test[y_test==0]
x_t1 = x_test[y_test==1]

#決定境界描画用x1の値からx2の値を計算する。
def b(x, w):
    return -(w[0] + w[1] * x)/ w[2]

#散布図のx1の最小値と最大値
xl = np.asarray([x[:,1].min(), x[:,1].max()])
yl = b(xl, w)
a1 = [xl[0], yl[0]]
a2 = [xl[1], yl[1]]
print('境界線の直線が通る2点\n', a1, a2)

plt.figure(figsize=(6, 6))
#散布図の表示
plt.scatter(x_t0[:,1], x_t0[:,2], marker='x', c='b', s=50, label='class 0')
plt.scatter(x_t1[:,1], x_t1[:,2], marker='o', c='k', s=50, label='class 1')

#散布図に決定境界の直線も追記
plt.plot(xl, yl, c='b')
plt.xlabel('sepal_length', fontsize=14)
plt.ylabel('sepal_width', fontsize=14)
plt.xticks(size=16)
plt.yticks(size=16)
plt.legend(fontsize=16)
plt.show()

#学習曲線の表示(損失関数)
plt.figure(figsize=(6, 4))
plt.plot(history[:,0], history[:,1], 'b')
plt.xlabel('iter', fontsize=14)
plt.ylabel('cost', fontsize=14)
plt.title('iter vs cost', fontsize=14)
plt.show()

#学習曲線の表示(精度)
plt.figure(figsize=(6, 4))
plt.plot(history[:,0], history[:,2], 'b')
plt.xlabel('iter', fontsize=14)
plt.ylabel('accuracy', fontsize=14)
plt.title('iter vs accuracy', fontsize=14)
plt.show()

x1 = np.linspace(4, 7.5, 100)
x2 = np.linspace(2, 4.5, 100)
xx1, xx2 = np.meshgrid(x1, x2)
xxx = np.asarray([np.ones(xx1.ravel().shape), xx1.ravel(), xx2.ravel()]).T
c = pred(xxx, w).reshape(xx1.shape)
plt.figure(figsize=(8,8))
ax = plt.subplot(1, 1, 1, projection='3d')
ax.plot_surface(xx1, xx2, c, color='blue',
    edgecolor='black', rstride=10, cstride=10, alpha=0.1)
ax.scatter(x_t1[:,1], x_t1[:,2], 1, s=20, alpha=0.9, marker='o', c='b')
ax.scatter(x_t0[:,1], x_t0[:,2], 0, s=20, alpha=0.9, marker='s', c='b')
ax.set_xlim(4,7.5)
ax.set_ylim(2,4.5)
ax.view_init(elev=20, azim=60)
