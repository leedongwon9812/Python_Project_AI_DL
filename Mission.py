import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'
path = "가공데이터.csv"
gdp = pd.read_csv(path, encoding="cp949")

# 변수 분류
independent = gdp[['수출 (100만달러)', '수입 (100만달러)', '장래인구 (1000명)', '실업률 (%)', '소비자물가지수']]
dependent = gdp[['1인당 GDP (달러)']]
year = gdp[['시점']]

# 모델 생성
X = tf.keras.layers.Input(shape=[5])
H = tf.keras.layers.BatchNormalization()(X)
H = tf.keras.layers.Activation('swish')(H)
H = tf.keras.layers.Dense(10)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('swish')(H)
H = tf.keras.layers.Dense(10)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('swish')(H)
H = tf.keras.layers.Dense(10)(H)
H = tf.keras.layers.Dense(10, activation='swish')(H)
H = tf.keras.layers.Dense(10, activation='swish')(H)
H = tf.keras.layers.Dense(10, activation='swish')(H)
Y = tf.keras.layers.Dense(1)(H)

model = tf.keras.models.Model(X,Y)
model.compile(loss='mse')
model.summary()

# 학습 ( y = w1x1 + w2x2 + ... + b )
model.fit(independent, dependent, epochs=2000, verbose=0)

# 모델 학습의 결과지 확인
# print(model.get_weights())

# 예측
print('예측값 (1990~1994)')
print(model.predict(independent[0:5]))
print('실제값 (1990~1994)')
print(dependent[0:5])
print('예측값 (2016~2020)')
print(model.predict(independent[-5:]))
print('실제값 (2016~2020)')
print(dependent[-5:])

# 데이터 시각화
xp = np.array(year.values[:,0])
yp = np.array(dependent.values[:,0])

yp2 = model.predict(independent[:])
yp2 = yp2.reshape(31)

# plt.plot(xp, yp, 'o--k', ms=5, mec='k', mfc='g', label='실제값') # 꺾은선그래프
plt.bar(xp, yp, width=1.0, label='실제값') # 막대그래프
plt.legend()
plt.plot(xp, yp2, 'o--k', ms=5, mec='k', mfc='r', label='예측값')
plt.legend()
plt.title('대한민국 1인당 GDP 변화 추이 (1990~2020)')
plt.show()