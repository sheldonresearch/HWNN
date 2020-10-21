#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@project:HWNN
@author:xiangguosun
@contact:sunxiangguodut@qq.com
@website:http://blog.csdn.net/github_36326955
@file: dblp_format.py
@platform: macOS High Sierra 10.13.1 Pycharm pro 2017.1
@time: 2019/10/16

16604个作者
第一步：doc2vec模型将author_interest文件转换为feature矩阵
第二部：Aminer_coauthor本身就是.cites文件
第三部：将author_class与feature拼接，构成.content文件
"""

import numpy as np
from collections import defaultdict
import torch
import random
import os
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#
all_cites=np.loadtxt('../pubmed.cites',delimiter='\t',dtype=str)

all_content=np.loadtxt('../pubmed.content',delimiter='\t',dtype=str)

rand_arr = np.arange(all_cites.shape[0])
np.random.shuffle(rand_arr)
slide_capacity=4000
cites=all_cites[rand_arr[0:slide_capacity]]

node_set=set()
for edge in cites:
    node_set.add(edge[0])
    node_set.add(edge[1])


print("node number:",len(list(node_set)))

content=[]

for row in all_content:
    node=row[0]
    if node in node_set:
        content.append(row)


np.savetxt('./pubmed.content', content, fmt='%s', delimiter='\t')

with open('./pubmed.cites','w') as fout:
    for edge in cites:
        src,des=edge[0],edge[1]
        fout.write(src+'\t'+des+'\n')