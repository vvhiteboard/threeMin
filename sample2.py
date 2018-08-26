import tensorflow as tf
import numpy as np

# 테스트 데이터
# x_data = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])
# y_data = np.array([
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [1, 0, 0],
#     [1, 0, 0],
#     [0, 0, 1]
# ])

data = np.loadtxt('./data.csv', delimiter=",", unpack=True, dtype="float32")

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

# 변수 선언
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 가중치 변수 생성
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.))

# 편향 변수 생성
b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))

# 첫번째 Layer(입력 -> 은닉)에 대한 오퍼레이션 정의
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

# 두번째 Layer(은닉 -> 출력)에 대한 오퍼레이션 정의
model = tf.add(tf.matmul(L1, W2), b2)
# model = tf.nn.softmax()

# 손실값에 대한 평균을 구함 (Y : 실제값, model : 예측값)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

# 최적화 함수 정의 (AdamOptimizer)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

### 학습 시작 ###
# 변수 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 학습
for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# 결과 도출 (예측값과 실제값 비교)
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print("예측값:", sess.run(prediction, feed_dict={X: x_data}))
print("실제값:", sess.run(target, feed_dict={Y: y_data}))

# 정확도 도출
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도: %.2f" % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
