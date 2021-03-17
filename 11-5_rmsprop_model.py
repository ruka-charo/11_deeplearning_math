# Macの問題回避
import os
import platform
if platform.system() == 'Darwin':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

# PDF出力用
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')


'''KerasによるNeural Network'''
# データ準備
# 変数定義

# D:入力ノード数
D = 784

# H:隠れ層のノード数
H = 128

#分類クラス数
num_classes = 10

# Kerasの関数でデータの読み込み
(x_train_org, y_train), (x_test_org, y_test) = mnist.load_data()

# 入力データの加工(次元を1次元に)
x_train = x_train_org.reshape(-1, D) / 255.0
x_test = x_test_org.reshape((-1, D)) / 255.0

# 正解データの加工(One Hot Vector)
y_train_ohe = np_utils.to_categorical(y_train, num_classes)
y_test_ohe = np_utils.to_categorical(y_test, num_classes)

# モデルの定義
# Sequentialモデルの定義
model = Sequential()

# 隠れ層1の定義
model.add(Dense(H, activation='relu', kernel_initializer='he_normal', input_shape=(D,)))

# 隠れ層2の定義
model.add(Dense(H, activation='relu', kernel_initializer='he_normal'))

# 出力層
model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))

# モデルのコンパイル
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop', metrics=['accuracy'])

# 学習
# 学習の単位
batch_size = 512

# 繰り返し回数
nb_epoch = 50

# モデルの学習
history1 = model.fit(
    x_train,
    y_train_ohe,
    batch_size = batch_size,
    epochs = nb_epoch,
    verbose = 1,
    validation_data = (x_test, y_test_ohe)
)
