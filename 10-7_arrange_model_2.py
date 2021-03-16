import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# PDF出力用
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')


'''データ読み込み'''
#データ読み込み
mnist = fetch_openml('mnist_784', version=1)
x_org, y_org = mnist.data, mnist.target.astype(np.int)

'''入力データ加工'''
#入力データ加工
#Step1 データ正規化 値の範囲を[0, 1]とする。
x_norm = x_org / 255.0

#先頭にダミー変数(1)を追加
x_all = np.insert(x_norm, 0, 1, axis=1)
print('ダミー変数追加後', x_all.shape)

#Step2 yをOne-hot-Vectorにする。
ohe = OneHotEncoder(sparse=False)
y_all_one = ohe.fit_transform(np.c_[y_org])
print('One Hot Vector化後', y_all_one.shape)

#Step3 学習データ、検証データに分割
x_train, x_test, y_train, y_test, y_train_one, y_test_one = train_test_split(
    x_all, y_org, y_all_one, train_size=60000, test_size=10000, shuffle=False)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape,
    y_train_one.shape, y_test_one.shape)

#データ内容の確認
N = 20
np.random.seed(12)
indexes = np.random.choice(y_test.shape[0], N, replace=False)
x_selected = x_test[indexes, 1:]
y_selected = y_test[indexes]
plt.figure(figsize=(10, 3))
for i in range(N):
    ax = plt.subplot(2, N/2, i + 1)
    plt.imshow(x_selected[i].reshape(28, 28), cmap='gray_r')
    ax.set_title('%d' %y_selected[i], fontsize=16)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

'''予測関数'''
#シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#softmax関数
def softmax(x):
    x_max = x.max(axis=0)
    x = x - x_max
    w = np.exp(x)
    return (w / w.sum(axis=0)).T

'''ReLU関数の導入'''
#ReLU関数
def ReLU(x):
    return np.maximum(0, x)

#step関数
def step(x):
    return 1.0 * (x > 0)

# ReLU関数とstep関数のグラフ表示
xx =  np.linspace(-4, 4, 501)
yy = ReLU(xx)
plt.figure(figsize=(6,6))
#plt.ylim(0.0, 1.0)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)
plt.grid(lw=2)
plt.plot(xx, ReLU(xx), c='b', label='ReLU', linestyle='-', lw=3)
plt.plot(xx, step(xx), c='k', label='step', linestyle='-.', lw=3)
plt.xticks(size=14)
plt.yticks(size=14)
plt.legend(fontsize=14)
plt.show()


'''評価関数'''
#交差エントロピー関数
def cross_entropy(yt, yp):
    return -np.mean(np.sum(yt * np.log(yp), axis=1))

#評価処理(戻り値は精度と損失関数)
def evaluate(x_test, y_test, y_test_one, U, V, W):
    b1_test = np.insert(ReLU(U @ x_test.T), 0, 1, axis=0)
    d1_test = np.insert(ReLU(V @ b1_test), 0, 1, axis=0)
    yp_test_one = softmax(W.T @ d1_test)
    yp_test = np.argmax(yp_test_one, axis=1)
    loss = cross_entropy(y_test_one, yp_test_one)
    score = accuracy_score(y_test, yp_test)
    return score, loss

'''ミニバッチ処理'''
class Indexes:
    #コンストラクタ
    def __init__(self, total, size):
        #配列全体の大きさ
        self.total = total
        #batchサイズ
        self.size = size
        #作業用indexes 初期値はNULLにしておく。
        self.indexes = np.zeros(0)

    #index取得関数
    def next_index(self):
        next_flag = False

        #batchサイズより作業用indexesが小さい場合はindexesを再生成
        if len(self.indexes) < self.size:
            self.indexes = np.random.choice(
                self.total, self.total,replace=False)
            next_flag = True

        #戻り用index取得と作業用indexesの更新
        index = self.indexes[:self.size]
        self.indexes = self.indexes[self.size:]
        return index, next_flag

#indexesクラスのテスト
#クラスの初期化
# 20:全体の配列の大きさ
# 5:一回に取得するindexの数
indexes = Indexes(20, 5)

for i in range(6):
    #next_index関数の呼び出し
    #戻り値1:indexのnumpy配列
    #戻り値2:作業用indexの更新があったかどうか
    arr, flag = indexes.next_index()
    print(arr, flag)


'''初期化処理　その1'''
#変数初期宣言　初期バージョン
#隠れ層のノード数
H = 128
H1 = H + 1
#M:訓練用系列データ総数
M = x_train.shape[0]
#D:入力データ次元数
D = x_train.shape[1]
#N:分類クラス数
N = y_train_one.shape[1]

#繰り返し回数
nb_epoch = 100
#ミニバッチサイズ
batch_size = 512
B = batch_size
#学習率
alpha = 0.01

#重み行列の初期設定(全て1)
#V = np.ones((H, D))
#W = np.ones((H1, N))
np.random.seed(123)
U = np.random.randn(H, D) / np.sqrt(D / 2)
V = np.random.randn(H, H1) / np.sqrt(H1 / 2)
W = np.random.randn(H1, N) / np.sqrt(H1 / 2)


#評価結果記録用(損失関数値と精度)
history1 = np.zeros((0, 3))

#ミニバッチ用関数初期化
indexes = Indexes(M, batch_size)

#繰り返し回数カウンタ初期化
epoch = 0


'''メイン処理'''
#メイン処理
while epoch < nb_epoch:
    #学習対象の選択(ミニバッチ学習法)
    index, next_flag = indexes.next_index()
    x, yt = x_train[index], y_train_one[index]

    #予測値計算(順伝播)
    a = U @ x.T
    b = ReLU(a)
    b1 = np.insert(b, 0, 1, axis=0)
    c = V @ b1
    d = ReLU(c)
    d1 = np.insert(d, 0, 1, axis=0)
    u = W.T @ d1
    yp = softmax(u)

    #誤差計算
    yd = yp - yt
    dd = step(c) * (yd @ W[1:].T).T
    bd = step(a) * (V[:,1:].T @ dd)

    #勾配計算
    W = W - alpha * (d1 @ yd) / B
    V = V - alpha * (dd @ b1.T) / B
    U = U - alpha * (bd @ x) / B

    # ログ記録用
    if next_flag: # 1 epoch 終了後の処理
        score, loss = evaluate(x_test, y_test, y_test_one, U, V, W)
        history1 = np.vstack((history1, np.array([epoch, loss, score])))
        print("epoch = %d loss = %f score = %f" % (epoch, loss, score))
        epoch = epoch + 1


'''結果確認　その1'''
#損失関数値と精度の確認
print('初期状態: 損失関数:%f 精度:%f' % (history1[0,1], history1[0,2]))
print('最終状態: 損失関数:%f 精度:%f' % (history1[-1,1], history1[-1,2]))

# 学習曲線の表示 (損失関数値)
plt.plot(history1[:,0], history1[:,1])
plt.ylim(0,2.5)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()

# 学習曲線の表示 (精度)
plt.plot(history1[:,0], history1[:,2])
plt.ylim(0,1)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()
