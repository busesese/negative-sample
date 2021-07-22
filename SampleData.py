# -*- coding: utf-8 -*-
# @Time    : 2020-12-17 10:28
# @Author  : WenYi
# @Contact : 1244058349@qq.com
# @Description :  script description

import random
import numpy as np
import pandas as pd
from tqdm import tqdm


class Sample(object):
    """
    正负样本采样，正样本标签必须为1，负样本标签为0
    """
    def __init__(self, data, col, target):
        """
        数据采样策略，给定一个样本集，用一定的采样策略进行采样
        :param data: DataFrame
        :param col: sample column name
        :param target: target column name
        """
        if data.empty:
            raise Exception("input data is empty you must input a not empty data")
        if not isinstance(col, str):
            raise Exception("input col must be str for sample column")
        if not isinstance(col, str):
            raise Exception("input col must be str for sample column")
        self.data = data.reset_index(drop=True)
        self.col = col
        self.target = target
        
    def frequency_count(self):
        """
        统计样本中指定采样col的出现频次
        :return: dict
        """
        frequency_dict = self.data[self.col].value_counts().to_dict()
        return frequency_dict
    
    def random_negative_sample(self, negative_number):
        """
        随机负采样策略，固定正负样本比率
        :param negative_number: int, how many negative samples should be sample for one positive sample
        :param n_works: int, multiprocessing for negative sample generate
        :return: DataFrame
        """
        # 入参检验
        if not isinstance(negative_number, int):
            raise TypeError("input arguments type must be int")
        
        # 随机采样的样本
        samples = self.data[self.col].unique().tolist()
        columns = self.data.columns
        # 需要重新生成的列名
        sub_columns = [col for col in columns if col != self.col and col != self.target]
        # 负采样
        sample_data = list()
        for idx, rows in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            sample_data.append(rows[sub_columns + [self.col, self.target]].tolist())
            negative_sample = random.sample(samples, negative_number)
            for s in negative_sample:
                sample_data.append(rows[sub_columns].tolist() + [s, 0])
        sample_data = pd.DataFrame(sample_data, columns=sub_columns + [self.col, self.target])
        return sample_data
    
    def weight_random_negative_sample(self, negative_number, alpha, t1):
        """
        加权随机负采样， 根据论文《Sample Optimization For Display Advertising》
        :param negative_number: int, how many negative samples should be sample for one positive sample
        :param alpha: int, sample frequency for distinguish Al and Ah
        :param t1: float, user parameter for sample
        :return:
        """
        frequency_dict = self.frequency_count()
        
        # 将样本分为两份 Al and Ah, 其中Al<alpha,  Ah>alpha
        Al, Ah = dict(), dict()
        sum_Al, sum_Ah = 0, 0
        for k, val in frequency_dict.items():
            if val <= alpha:
                Al[k] = val
                sum_Al += val ** t1
            else:
                Ah[k] = val
                sum_Ah += val ** t1

        # AL中的样本，AH中的样本，以及AH中各个样本采样概率,概率计算公式见论文
        pl_sample = [k for k, _ in Al.items()]
        ph_sample, ph_sample_ratio = list(), list()
        for k, val in Ah.items():
            ph_sample.append(k)
            ph_sample_ratio.append(val**t1)
        ph_sample_ratio = [val/sum(ph_sample_ratio) for val in ph_sample_ratio]
        
        # 计算pl
        pl = sum_Al / (sum_Al + sum_Ah)
        
        # 样本采样
        columns = self.data.columns
        # 需要重新生成的列名
        sub_columns = [col for col in columns if col != self.col and col != self.target]
        
        sample_data = list()
        for idx, rows in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            sample_data.append(rows[sub_columns + [self.col, self.target]].tolist())
            # 负采样，对于每次采样负样本，首先确定是在Al中采还是Ah中采，Al中用随机采样，Ah中用一元分布采样
            # 这样做的目的是保证低频样本被采样的次数增多(相对于Ah中的采样方式来说)
            for i in range(negative_number):
                if random.random() < pl:
                    # 均匀采样
                    s = random.sample(pl_sample, 1)[0]
                    sample_data.append(rows[sub_columns].tolist() + [s, 0])
                else:
                    # word2vec 采样
                    s = np.random.choice(a=ph_sample, size=1, p=ph_sample_ratio)[0]
                    sample_data.append(rows[sub_columns].tolist() + [s, 0])
        sample_data = pd.DataFrame(sample_data, columns=sub_columns + [self.col, self.target])
        return sample_data


if __name__ == "__main__":
    
    import time
    start_time = time.time()
    df = pd.read_csv('example_data/sample_data.csv')
    data = Sample(df, 'item', 'target')
    df = data.weight_random_negative_sample(5, 10, 0.75)
    df.to_csv('weight_sample.csv', index=False)
    print(time.time() - start_time)
    start_time = time.time()
    df = data.random_negative_sample(5)
    df.to_csv('weight_sample.csv', index=False)
    print(time.time() - start_time)
