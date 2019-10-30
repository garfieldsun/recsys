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

def softmax(x):
    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        fenzi_exp= tf.exp(x - tf.reduce_max(x))
        fenmu = 1.0/tf.reduce_sum(fenzi_exp,axis=1,keepdims=True)
        result = fenzi_exp * fenmu
    else:
        # Vector
        ### YOUR CODE HERE
        result = tf.exp(x - tf.reduce_max(x))/tf.reduce_sum(tf.exp(x - tf.reduce_max(x)))
        ### END YOUR CODE
    return result


def AFM(xi,xv,y,xi_test,xv_test,y_test,feature_size,field_size,embedding_size=16,attention_size=10,
        lr = 0.001,epoch=100):
    '''
    1权重初始化
    2Embedding
    3模型
    4训练
    :return:
    '''
    '''1、权重初始化分为FM部分和Deep部分'''

    #FM权重
    w_0 = tf.Variable(tf.constant(0.1),name='bias')
    w = tf.Variable(tf.random.normal([feature_size, 1], mean=0, stddev=0.01),name='first_weight')
    v = tf.Variable(tf.random.normal([feature_size, embedding_size], mean=0, stddev=0.01),name='second_weight')

    #Attention部分的权重
    weights={}


    glorot = np.sqrt(2.0 / (attention_size + embedding_size))

    weights['attention_w'] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(embedding_size,attention_size)), dtype=np.float32
    )
    weights['attention_b'] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(attention_size,)), dtype=np.float32
    )
    weights['attention_h'] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(attention_size,)), dtype=np.float32
    )
    weights['attention_p'] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(embedding_size,)), dtype=np.float32
    )



    '''2、Embedding'''
    feat_index = tf.compat.v1.placeholder(tf.int32,[None,None],name='feat_index')
    feat_value = tf.compat.v1.placeholder(tf.float32,[None,None],name='feat_value')
    label = tf.compat.v1.placeholder(tf.float32,shape=[None,1],name='label')
    # dropout_keep_deep = tf.placeholder(tf.float32,shape=[None],name='dropout_deep_deep')


    embedding_first =tf.nn.embedding_lookup(w,feat_index)   #None*F *1   F是field_size大小，也就是不同域的个数
    embedding = tf.nn.embedding_lookup(v,feat_index)      #None * F * embedding_size
    feat_val = tf.reshape(feat_value,[-1,field_size,1])
    '''3、模型'''
    # first_order term +偏置
    y_first_order= tf.reduce_sum(tf.multiply(embedding_first,feat_val),2)  # None*F
    y_first_order_num = tf.reduce_sum(y_first_order,1,keepdims=True)    # None*1
    liner = tf.add(y_first_order_num, w_0)  # None*1

    # 交互项的矩阵 #None*F(F-1)/2*k
    attention_weight_list=[]
    for i in range(field_size):
        for j in range(i+1,field_size):
            attention_weight_list.append(tf.multiply(embedding[:,i,:],embedding[:,j,:]))
    attention_cross = tf.stack(attention_weight_list)
    attention_cross = tf.transpose(attention_cross,perm=[1,0,2],name='transpose')   #None*F(F-1)/2*k

    # 可以带入权重attention
    attention_weight = tf.add(tf.matmul(attention_cross,weights['attention_w']),weights['attention_b'])  #None*F(F-1)/2*A
    attention_weight_a = tf.reduce_sum(tf.multiply(tf.nn.relu(attention_weight),weights['attention_h']),axis=2)  #None*F(F-1)/2
    # b=softmax(attention_weight_a)

    attention_weight_out = tf.reshape(softmax(attention_weight_a),[-1,int(field_size*(field_size-1)/2),1]) #None*F(F-1)/2*1

    #二阶second_order的输出
    #所有交叉项相加
    atten_add = tf.reduce_sum(tf.multiply(attention_cross,attention_weight_out),axis=1)     #N*K
    second_order = tf.reduce_sum(tf.multiply(atten_add,weights['attention_p']),axis=1,keepdims=True)  #N*1
#
#
# 输出
    out = tf.add(liner,second_order)  #N*1

    #loss
    loss = tf.nn.l2_loss(tf.subtract(out,label))
    optimizer = tf.compat.v1.train.AdamOptimizer(lr,beta1=0.9,beta2=0.999,epsilon=1e-8).minimize(loss)

    # init
    init = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session()

    sess.run(init)

    '''训练'''
    batch_size = 256
    num_batch = len(xi)//batch_size + 1
    for step in range(epoch):

        losses=[]
        for i in range(num_batch):
            idmin = i* batch_size
            idmax = np.min([(i+1)*batch_size,len(xi)])
            xi_batch = xi[idmin:idmax]
            xv_batch = xv[idmin:idmax]
            y_batch = y[idmin:idmax]
            tmp_loss,_ = sess.run([loss,optimizer],feed_dict={feat_index:xi_batch,feat_value:xv_batch,
                                                        label:y_batch})
            losses.append(tmp_loss)
        nnloss = np.mean(losses)
        print('epoch:%d loss:%f'%(step,nnloss))

    vali_loss = sess.run(loss,feed_dict={feat_index:xi_test,feat_value:xv_test,label:y_test})
    print(vali_loss)

    predict = sess.run(out,feed_dict={feat_index:xi_test,feat_value:xv_test,label:y_test})
    rmse = np.sqrt(np.mean(np.square(predict-y_test)))
    print(rmse)




if __name__=="__main__":
    df = pd.read_csv('D:/huashuData/用户特征/dianbo_user_3.csv', converters={'user_id': str, 'program_id': str})
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['user_id'], random_state=10)
    num_col = ['count']
    label_col = ['rate']
    dict, feature_size = get_feature_dict(df, num_col)
    x_train_index, x_train_value, y_train = get_data(df_train, dict)
    x_test_index, x_test_value, y_test = get_data(df_test, dict)
    m,field_size  = np.array(x_train_index).shape


    AFM(x_train_index,x_train_value,y_train,x_test_index,x_test_value,y_test,feature_size,field_size)