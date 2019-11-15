#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
from prodata import dataSet
import math
import heapq
from time import time

class NCF():
    def __init__(self,file_path,
                 embedding_size=8,
                 deep_layers=[64,32,16,8], dropout_deep=[0.5,0.5,0.5,0.5],
                 activation = tf.nn.relu,
                 l2_reg=0.01,topK=10,
                 epoch=50, batch_size=256,
                 learning_rate=0.001,random_seed=2019):
        self.dataSet = dataSet(file_path)
        self.user_train = self.dataSet.user_train
        self.item_train = self.dataSet.item_train
        self.label_train = self.dataSet.click

        self.user_test = self.dataSet.user_neg
        self.item_test = self.dataSet.item_neg

        self.user_size = self.dataSet.user_size
        self.item_size = self.dataSet.item_size

        self.testUser,self.testItem=self.testData()

        self.embedding_size = embedding_size
        self.topK = topK

        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = activation
        self.l2_reg = l2_reg
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self._init_graph()

    def _init_graph(self):
        self.user = tf.placeholder(tf.int32,name='user')
        self.item = tf.placeholder(tf.int32,name='item')
        self.label = tf.placeholder(tf.float32,name="label")

        weights={}
        weights['GMF_user'] = tf.Variable(tf.random_normal([self.user_size, self.embedding_size], 0.0, 0.01),name="gmf_user")
        weights['GMF_item'] = tf.Variable(tf.random_normal([self.user_size, self.embedding_size], 0.0, 0.01),name='gmf_item')

        weights['MLP_user'] = tf.Variable(tf.random_normal([self.user_size, int(self.deep_layers[0]/2)], 0.0, 0.01),
                                        name='MLP_user')
        weights['MLP_item'] = tf.Variable(tf.random_normal([self.user_size, int(self.deep_layers[0]/2)], 0.0, 0.01),
                                          name='MLP_item')

        #deep layer的权重
        num_layer = len(self.deep_layers)
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]
            input_size = self.embedding_size + self.deep_layers[-1]
            glorot = np.sqrt(2.0 / (input_size + 1))
            #最后一层的权重
            weights["concat_pro"] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                dtype=np.float32)  # layers[i-1]*layers[i]
            weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)


        #---------------------------MF part--------------------------------------------------------
        gmf_user_embedding = tf.nn.embedding_lookup(weights['GMF_user'],self.user)
        gmf_item_embedding = tf.nn.embedding_lookup(weights['GMF_item'],self.item)

        self.gmf_vector = tf.multiply(gmf_user_embedding,gmf_item_embedding)       #None * embedding_size
        #---------------------------MLP part-------------------------------------------------------

        mlp_user_embedding = tf.nn.embedding_lookup(weights['MLP_user'],self.user)
        mlp_item_embedding = tf.nn.embedding_lookup(weights['MLP_item'],self.item)
        mlp_tmp_vector = tf.concat([mlp_user_embedding,mlp_item_embedding],axis=1,name='interaction')   #None * Layer[0]
        self.y_deep = mlp_tmp_vector
        for i in range(1, len(self.deep_layers)):
            self.y_deep = tf.add(tf.matmul(self.y_deep, weights["layer_%d" % i]),
                                 weights["bias_%d" % i])  # None * layer[i] * 1
            self.y_deep = self.deep_layers_activation(self.y_deep)
            self.mlp_vector = tf.nn.dropout(self.y_deep, self.dropout_deep[i])  # None *deep_layer[-1]


        #---concat层
        
        concat_input = tf.concat([self.gmf_vector,self.mlp_vector],axis=1)
        self.out = tf.add(tf.matmul(concat_input, weights["concat_pro"]), weights["concat_bias"])
        self.out = tf.nn.sigmoid(self.out)


        # loss
        self.loss = tf.losses.log_loss(self.label, self.out)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                epsilon=1e-8).minimize(self.loss)

        # init
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess = self._init_session()
        self.sess.run(init)

    # def pp(self,Xi,Xv,y):
    #     feed_dict = {self.user: Xi ,
    #                  self.item: Xv,
    #                  self.label:y
    #                  # self.label: y_batch,
    #                  }
    #     # label= self.sess.run(y)
    #     batch_out = self.sess.run(self.loss, feed_dict=feed_dict)
    #     return batch_out

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def get_batch(self,xuser,xitem,y,index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        end = end if end < len(y) else len(y)
        return xuser[start:end],xitem[start:end],y[start:end]

    def fit_on_batch(self, x_train, i_train, y_train):
        loss, opt = self.sess.run([self.loss, self.optimizer], feed_dict={self.user:x_train,
                                                                          self.item: i_train,
                                                                          self.label:y_train})
        return loss

    def fit(self):
        best_hr = -1
        best_ndcg = -1
        best_epoch = -1
        print("Start Training!")
        total_batch = int(len(self.label_train) / self.batch_size)
        for step in range(self.epoch):
            losses = []
            t1 = time()
            for i in range(total_batch):
                xuser_batch, xitem_batch, y_batch = self.get_batch(self.user_train,self.item_train,self.label_train,i)
                loss = self.fit_on_batch(xuser_batch, xitem_batch, y_batch)
                losses.append(loss)
            all_loss = np.mean(losses)
            t2 = time()
            hr, ndcg = self.evaluate(self.topK)
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (step, t2 - t1, hr, ndcg, all_loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_epoch = hr, ndcg, step
        print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_epoch, best_hr, best_ndcg))

    def predict(self,xuser,xitem):
        dummy_y = [1] * len(xuser)
        y_pred = []
        batch_index = 0
        Xi_batch, Xv_batch, y_batch= self.get_batch(xuser,xitem,dummy_y,batch_index)
        # y_pred = None
        while len(Xi_batch) > 0:
            # num_batch = len(y_batch)
            feed_dict = {self.user: Xi_batch,
                         self.item: Xv_batch
                         # self.label: y_batch,
                         }
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)
            y_pred.extend(batch_out)

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(xuser,xitem,dummy_y,batch_index)

        y_pred = np.array(y_pred).reshape(-1)
        return y_pred

    def testData(self):
        '''
        根据predict生成测试数据
        :return:
        '''

        user = np.array(self.user_test).reshape(-1, 100)
        item = np.array(self.item_test).reshape(-1, 100)
        return user, item

    def evaluate(self, topK):
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0

        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i + 2)
            return 0

        hr = []
        NDCG = []
        testUser = self.testUser
        testItem = self.testItem
        testRate = self.predict(self.user_test,self.item_test).reshape(-1, 100)
        for i in range(len(testUser)):
            target = testItem[i][0]
            item_score_dict = {}
            for j in range(len(testItem[i])):
                item = testItem[i][j]
                item_score_dict[item] = testRate[i][j]

            ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr.append(tmp_hr)
            NDCG.append(tmp_NDCG)
        return np.mean(hr), np.mean(NDCG)

if __name__ == "__main__":
    file_path = './Data/dianbo_user_pro_3.csv'
    model = NCF(file_path)
    model.fit()

        
        












