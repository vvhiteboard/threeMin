import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

# 균등분포에서 무작위로 데이터를 추출함 (random_uniform)
# 정규분포로 생성하고 싶으면 random_normal 메소드를 사용하면 됨
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# 텐서플로 플레이스홀더를 선언 (placeholder)
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# 수식 작성
hypothesis = W * X + b

# 실제값과 예측값 차의 제곱의 평균을 구해 손실값(cost)을 계산한다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y:y_data})
        
        print(step, cost_val, sess.run(W), sess.run(b))

    print("\n======== Test ==========")
    print("X: 5, Y: ", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, Y: ", sess.run(hypothesis, feed_dict={X: 2.5}))

