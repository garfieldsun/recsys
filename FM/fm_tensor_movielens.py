#-*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

cols = ['user', 'item', 'rating', 'timestamp']

train = pd.read_csv('./ua.base', delimiter='\t', names=cols,
                    converters={'user': str, 'item': str})
test = pd.read_csv('./ua.test', delimiter='\t', names=cols,
                   converters={'user': str, 'item': str})
train = train.drop(['timestamp'], axis=1)
test = test.drop(['timestamp'], axis=1)

df_x_train = train[['user', 'item']]
df_x_test = test[['user', 'item']]
df_zong = pd.concat([df_x_train, df_x_test])
y_train = train['rating'].values.reshape(-1, 1)
y_test = test['rating'].values.reshape(-1, 1)
vec = DictVectorizer()
vec.fit_transform(df_zong.to_dict(orient='record'))
del df_zong
x_train = vec.transform(df_x_train.to_dict(orient='record')).toarray()
x_test = vec.transform(df_x_test.to_dict(orient='record')).toarray()
print("x_train shape", x_train.shape)
print("x_test shape", x_test.shape)

k = 8
learning_rate = 0.05
l2_w = 0.001
l2_v = 0.001
epoch = 100

m, n = x_train.shape
print(m, n)
x = tf.placeholder(tf.float32, [None, n], name='feature')
y = tf.placeholder(tf.float32, [None, 1], name='label')
'''初始化参数'''
w_0 = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.random_normal([n], mean=0, stddev=0.01))
v = tf.Variable(tf.random_normal([k, n], mean=0, stddev=0.01))
'''前向传播'''
# ---------first order---------
first_order = tf.reduce_sum(tf.multiply(w, x), axis=1, keepdims=True)
liner = tf.add(w_0, first_order)  # None*1
# ---------second order---------
mul_square = tf.pow(tf.matmul(x, tf.transpose(v)), 2)  # None * n
squqre_mul = tf.matmul(tf.pow(x, 2), tf.transpose(tf.pow(v, 2)))  # None * n
second_order = 0.5 * tf.reduce_sum(tf.subtract(mul_square, squqre_mul), 1, keepdims=True)  # None * 1
# ---------FM Model的预测值---------
fm_model = tf.add(liner, second_order)  # None * 1

'''损失函数'''
loss = tf.reduce_mean(tf.square(fm_model - y))
l2_norm = tf.reduce_sum(l2_w * tf.square(w) + l2_v * tf.square(v))
losses = tf.add(loss, l2_norm)

'''随机梯度下降'''
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(losses)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for step in range(epoch):
        tmp_loss, _ = sess.run([losses, train_op], feed_dict={x: x_train, y: y_train})

        validate_loss = sess.run(loss, feed_dict={x: x_test, y: y_test})
        print("epoch:%d train loss:%f validate_loss:%f" %
              (step, tmp_loss, validate_loss))

        rmse = sess.run(loss, feed_dict={x: x_test, y: y_test})
        RMSE = np.sqrt(rmse)
        print("rmse:", RMSE)

    predict = sess.run(fm_model, feed_dict={x: x_test})
    print("predict:", predict)
