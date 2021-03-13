import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from mpl_toolkits.mplot3d import Axes3D

# PDF出力用
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')


'''データ読み込み'''
#学習用データ準備
iris = load_iris()
x_org, y_org = iris.data, iris.target

#入力データに関しては、sepal_length(0)とpetal_length(2)のみ抽出
x_select = x_org[:,[0,2]]
print('元データ', x_select.shape, y_org.shape)


'''学習データの散布図表示'''
#散布図の表示
x_t0 = x_select[y_org == 0]
x_t1 = x_select[y_org == 1]
x_t2 = x_select[y_org == 2]
plt.figure(figsize=(6,6))
plt.scatter(x_t0[:,0], x_t0[:,1], marker='x', c='k', s=50, label='0(setosa)')
plt.scatter(x_t1[:,0], x_t1[:,1], marker='o', c='b', s=50, label='1(versicolour)')
plt.scatter(x_t2[:,0], x_t2[:,1], marker='+', c='k', s=50, label='2(virginica)')
plt.xlabel('sepal_length', fontsize=14)
plt.ylabel('petal_length', fontsize=14)
plt.xticks(size=14)
plt.yticks(size=14)
plt.legend(fontsize=14)
plt.show()


'''データ前処理'''
#ダミー変数を追加
x_all2 = np.insert(x_org, 0, 1.0, axis=1)

#yをOne-hot-Vectorにする。
ohe = OneHotEncoder(sparse=False, categories='auto')
y_work = np.c_[y_org]
y_all_one = ohe.fit_transform(y_work)
print('オリジナル', y_org.shape)
print('2次元化', y_work.shape)
print('One Hot Vector化後', y_all_one.shape)

#学習データ、検証データに分析
x_train2, x_test2, y_train, y_test, \
y_train_one, y_test_one = train_test_split(x_all2, y_org, y_all_one,
    train_size=75, test_size=75, random_state=123)
print(x_train2.shape, x_test2.shape, y_train.shape,
    y_test.shape, y_train_one.shape, y_test_one.shape)

print('入力データ(x)')
print(x_train2[:5,:])
print('正解データ(y)')
print(y_train[:5])
print('正解データ(One Hot Vector化後)')
print(y_train_one[:5,:])

#学習対象の選択
x, yt, x_test = x_train2, y_train_one, x_test2

'''予測関数'''
#softmax関数(9.7.3)
def softmax(x):
    x = x.T
    x_max = x.max(axis=0)
    x = x - x_max
    w = np.exp(x)
    return (w / w.sum(axis=0)).T

#予測値の計算(9.7.1、9.7.2)
def pred(x, W):
    return softmax(x @ W)


'''評価'''
#交差エントロピー関数(9.5.1)
def cross_entropy(yt, yp):
    return -np.mean(np.sum(yt * np.log(yp), axis=1))

#モデル評価を行う関数
def evaluate(x_test, y_test, y_test_one, W):
    #予測値の計算(確率値)
    yp_test_one = pred(x_test, W)
    #確率値から予測クラスを導出
    yp_test = np.argmax(yp_test_one, axis=1)
    #損失関数値の計算
    loss = cross_entropy(y_test_one, yp_test_one)
    #精度の算出
    score = accuracy_score(y_test, yp_test)
    return loss, score


'''初期化処理'''
#初期化処理
#標本数
M = x.shape[0]
#入力次元数(ダミー変数を含む)
D = x.shape[1]
#分類先クラス数
N = yt.shape[1]

#繰り返し回数
iters = 10000

#学習率
alpha = 0.01

#重み行列の初期設定(全て1)
W = np.ones((D, N))

#評価結果記録用
history = np.zeros((0, 3))


#メイン処理(4次元版)
for k in range(iters):
    #予測値の計算(9.7.1)
    yp = pred(x, W)
    #誤差の計算(9.7.4)
    yd = yp - yt
    #重みの更新(9.7.5)
    W = W - alpha * (x.T @ yd) / M

    if k % 10 == 0:
        loss, score = evaluate(x_test, y_test, y_test_one, W)
        history = np.vstack((history, np.array([k, loss, score])))
        print('epoch = %d loss = %f score = %f' % (k, loss, score))

print(history.shape)


#損失関数値と精度の確認
print('初期状態：損失関数：%f 精度：%f' % (history[0, 1], history[0, 2]))
print('最終状態：損失関数：%f 精度：%f' % (history[-1, 1], history[-1, 2]))

#学習曲線の表示(損失関数)
plt.plot(history[:,0], history[:,1])
plt.ylim(0, 1.2)
plt.grid()
plt.xlabel('iter', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.title('iter vs loss', fontsize=14)
plt.show()

# 学習曲線の表示 (精度)
plt.plot(history[:,0], history[:,2])
plt.ylim(0,1)
plt.grid()
plt.xlabel('iter', fontsize=14)
plt.ylabel('accuracy', fontsize=14)
plt.title('iter vs accuracy', fontsize=14)
plt.show()
