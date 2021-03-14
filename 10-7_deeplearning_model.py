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
    x = x.T
    x_max = x.max(axis=0)
    x = x - x_max
    w = np.exp(x)
    return (w / w.sum(axis=0)).T

'''評価関数'''
#交差エントロピー関数
def cross_entropy(yt, yp):
    return -np.mean(np.sum(yt * np.log(yp), axis=1))

#評価処理(戻り値は精度と損失関数)
def evaluate(x_test, y_test, y_test_one, V, W):
    b1_test = np.insert(sigmoid(x_test, V), 0, 1, axis=1)
    yp_test_one = softmax(b1_test_one @ W)
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
indexes = Index(20, 5)

for i in range(6):
    #next_index関数の呼び出し
    #戻り値1:indexのnumpy配列
    #戻り値2:作業用indexの更新があったかどうか
    arr, flag = indexes.next_index()
    print(arr, flag)
