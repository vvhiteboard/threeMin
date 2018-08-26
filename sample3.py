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
global_step = tf.Variable(0, trainable=False, name="global_step")

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

# 변수 선언
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 가중치 변수 생성
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))

# 편향 변수 생성
b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([20]))
b3 = tf.Variable(tf.zeros([3]))

# 첫번째 Layer(입력 -> 은닉)에 대한 오퍼레이션 정의
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

# 두번째 Layer(은닉 -> 출력)에 대한 오퍼레이션 정의
L2 = tf.add(tf.matmul(L1, W2), b2)
L2 = tf.nn.relu(L2)

model = tf.add(tf.matmul(L2, W3), b3)

# 손실값에 대한 평균을 구함 (Y : 실제값, model : 예측값)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

# 최적화 함수 정의 (AdamOptimizer)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)

### 학습 시작 ###
# 변수 초기화
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

check = tf.train.get_checkpoint_state("./model")
if check and tf.train.checkpoint_exists(check.model_checkpoint_path):
    saver.restore(sess, check.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

# 학습
for step in range(2):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    print("Step : %d, " % sess.run(global_step),
          "Cost : %.3f" % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# 결과 저장
saver.save(sess, './model/dnn.check', global_step=global_step)


# 결과 도출 (예측값과 실제값 비교)
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print("예측값:", sess.run(prediction, feed_dict={X: x_data}))
print("실제값:", sess.run(target, feed_dict={Y: y_data}))

# 정확도 도출
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도: %.2f" % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
