import tensorflow as tf
import numpy as np

data = np.loadtxt('./data.csv', delimiter=",", unpack=True, dtype="float32")
global_step = tf.Variable(0, trainable=False, name="global_step")

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

# 변수 선언
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.name_scope("layer1"):
    W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
    b1 = tf.Variable(tf.zeros([10]))
    L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
    tf.summary.histogram("Weights1", W1)
    tf.summary.histogram("bias1", b1)

with tf.name_scope("layer2"):
    W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
    b2 = tf.Variable(tf.zeros([20]))
    L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
    tf.summary.histogram("Weights2", W2)
    tf.summary.histogram("bia2", b2)

with tf.name_scope("output"):
    W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
    b3 = tf.Variable(tf.zeros([3]))
    model = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))
    tf.summary.histogram("Weights3", W3)
    tf.summary.histogram("bia3", b3)

with tf.name_scope("optimizer"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step=global_step)
    tf.summary.scalar("cost", cost)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

check = tf.train.get_checkpoint_state("./model")
if check and tf.train.checkpoint_exists(check.model_checkpoint_path):
    saver.restore(sess, check.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs", sess.graph)

# 학습
for step in range(1000):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    print("Step : %d, " % sess.run(global_step),
          "Cost : %.3f" % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
    writer.add_summary(summary, global_step=sess.run(global_step))

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
