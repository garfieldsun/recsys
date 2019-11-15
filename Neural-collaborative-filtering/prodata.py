#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


class dataSet():
    def __init__(self, file_path ):
        self.file_path = file_path
        self.df, self.dfTrain, self.dfTest, self.user_size,self.item_size,self.user_inter= self.get_data()
        self.user_train,self.item_train,self.click = self.train_data_mf(4)
        self.user_neg, self.item_neg = self.test_data_mf(99)



    def get_data(self):
        '''

        :return:训练数据和测试数据
        '''
        df = pd.read_csv(self.file_path)
        del df['count']
        del df['rate']
        user_set= df['user_id'].unique()
        item_set=df['program_id'].unique()
        user_size=len(user_set)
        item_size=len(item_set)
        user_length = df.groupby('user_id').size().tolist()
        split_train_test=[]
        for i in user_set:
            for j in range(user_length[i]-1):
                split_train_test.append('train')
            split_train_test.append('test')
        df['split']=split_train_test
        dfTrain = df[df['split']=='train'].reset_index(drop=True)
        dfTest = df[df['split']=='test'].reset_index(drop=True)
        del df['split']
        del dfTrain['split']
        del dfTest['split']
        # print(df)
        # field_dim = len(df.columns)
        print('训练数据集长度：{}'.format(dfTrain.shape))
        print('测试数据集长度：{}'.format(dfTest.shape))
        print('用户数量：{}'.format(user_size))
        print('节目数量：{}'.format(item_size))
        #用户交互过的物品
        user_bought={}
        for i in range(len(df)):
            u=df['user_id'][i]
            j=df['program_id'][i]
            # l = 1
            user_bought[(u,j)]=1.0
        return df, dfTrain, dfTest,user_size,item_size,user_bought

    def train_data_mf(self,negNum):
        '''

        :return: MF算法中训练的训练数据
        '''
        user=[]
        item=[]
        click=[]

        # label=[]
        df_mf = self.dfTrain.copy()
        # print(df_mf)
        for i in range(len(df_mf)):
            u=df_mf['user_id'][i]
            j=df_mf['program_id'][i]
            # r=df_mf['rate'][i]
            user.append(u)
            item.append(j)
            click.append([1.0])
            for step in range(negNum):
                t = np.random.randint(self.item_size-1)
                while (u, t) in self.user_inter:
                    t = np.random.randint(self.item_size-1)
                user.append(u)
                item.append(t)
                click.append([0.0])
            # label.append(r)

        return user,item,click

    def test_data_mf(self,negNum):
        '''

        :return: 用户和项目
        '''
        user = []
        item = []
        df_test_mf = self.dfTest.copy()
        # print(df_mf)
        for i in range(len(df_test_mf)):
            u = df_test_mf['user_id'][i]
            j = df_test_mf['program_id'][i]
            user.append(u)
            item.append(j)
            neglist=set()
            neglist.add(j)
            for step in range(negNum):
                t = np.random.randint(self.item_size-1)
                while (u,t) in self.user_inter or t in neglist:
                    t = np.random.randint(self.item_size-1)
                neglist.add(t)
                user.append(u)
                item.append(t)
        return user, item


# if __name__ == "__main__":
#     file_path = 'D:/huashuData/用户特征/dianbo_user_pro_3.csv'
#     # ignore = ['rate']
#     # numeric = ['count']
#     f = dataSet(file_path)
#
#     # print(f.dfTrain)
#     # user,item,df_test_neg = f.test_data_mf(99)
#     # print(f.train_data_fm())
#
#     tt = f.click
#     print(tt)
#     print(len(tt))



