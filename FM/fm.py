#-*- coding:utf-8 -*-



import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_feature_dict(df,num_col):
    '''
    特征向量字典，其格式为{field：{特征：编号}}
    :param df:
    :return: {field：{特征：编号}}
    '''
    feature_dict={}
    total_feature=0
    df.drop('rate',axis=1,inplace=True)
    for col in df.columns:
        if col in num_col:
            feature_dict[col]=total_feature
            total_feature += 1
        else:
            unique_feature = df[col].unique()
            feature_dict[col]=dict(zip(unique_feature,range(total_feature,total_feature+len(unique_feature))))
            total_feature += len(unique_feature)
    return feature_dict,total_feature

def get_data(df,feature_dict):
    '''

    :param df:
    :return:
    '''
    y = df[['rate']].values
    dd = df.drop('rate',axis=1)
    df_index = dd.copy()
    df_value = dd.copy()
    for col in df_index.columns:
        if col in num_col:
            df_index[col] = feature_dict[col]
        else:
            df_index[col] = df_index[col].map(feature_dict[col])
            df_value[col] = 1.0
    xi=df_index.values.tolist()
    xv=df_value.values.tolist()
    return xi,xv,y





df = pd.read_csv('D:/huashuData/用户特征/dianbo_user_3.csv', converters={'user_id': str, 'program_id': str})
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['user_id'], random_state=10)
num_col = ['count']
label_col = ['rate']
dict,feature_size = get_feature_dict(df,num_col)
x_train_index,x_train_value,y_train = get_data(df_train,dict)
x_test_index, x_test_value, y_test = get_data(df_train, dict)
# print(y_train)

embedding_size = 8
lr = 0.02
l2_w = 0.01
l2_v = 0.01
epoch = 20
m, n = np.array(x_train_index).shape
print(m, n)
xidx = tf.placeholder(tf.int32, [None, None], name='feat_index')
xval = tf.placeholder(tf.float32, [None, None], name='feat_value')
y = tf.placeholder(tf.float32, [None, 1], name='label')
'''初始化参数'''
w_0 = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.random_normal([feature_size, 1], mean=0, stddev=0.01))
v = tf.Variable(tf.random_normal([feature_size, embedding_size], mean=0, stddev=0.01))
'''embedding'''
embedding_first = tf.nn.embedding_lookup(w, xidx)
embedding = tf.nn.embedding_lookup(v, xidx)  # N n embedding_size
value = tf.reshape(xval, [-1, n, 1])
embedding_value = tf.multiply(embedding, value)  # N n embedding_size
'''前向传播'''
# ---------first order---------
first_order = tf.reduce_sum(tf.multiply(embedding_first, value), axis=1)
liner = tf.add(w_0,first_order)  # None*1
# ---------second order---------
mul_square = tf.square(tf.reduce_sum(embedding_value, 1))# None * 8
squqre_mul = tf.reduce_sum(tf.square(embedding_value), 1)  # None * 8
second_order = 0.5 * tf.reduce_sum(tf.subtract(mul_square, squqre_mul), 1,keepdims=True)  # None * 1
# ---------FM Model的预测值---------
fm_model = tf.add(liner, second_order)  # None * 1

'''损失函数'''
loss = tf.reduce_mean(tf.square(fm_model - y))
# l2_norm = tf.reduce_sum(l2_w * tf.square(w) + l2_v * tf.square(v))
# losses = tf.add(loss, l2_norm)

'''随机梯度下降'''
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # second = sess.run(fm_model, feed_dict={xidx: x_train_index, xval: x_train_value, y: y_train})
    # print(second)
    for step in range(epoch):

        tmp_loss, _ = sess.run([loss, train_op], feed_dict={xidx: x_train_index, xval: x_train_value, y: y_train})
        print("epoch:%d tmp_loss：%f" %(step,tmp_loss))





# with tf.Session() as sess:
#     sess.run(init)
#
#     for step in range(epoch):
#         tmp_loss, _ = sess.run([losses, train_op], feed_dict={x: x_train, y: y_train})
#
#         validate_loss = sess.run(loss, feed_dict={x: x_test, y: y_test})
#         print("epoch:%d train loss:%f validate_loss:%f" %
#               (step, tmp_loss, validate_loss))
#
#         rmse = sess.run(loss, feed_dict={x: x_test, y: y_test})
#         RMSE = np.sqrt(rmse)
#         print("rmse:", RMSE)

