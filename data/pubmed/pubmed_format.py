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


"""
import numpy as np
from collections import defaultdict
import torch

col_name_list=[]
word_ind_map=defaultdict(int)
with open('./raw_content.txt','r') as fin:
    lines=fin.readlines()
    first_line=lines[1]
    col_name=first_line.split('\t')
    word_ind=0

    for i in range(1,len(col_name)-1):
        word=col_name[i].split(':')[1]
        word_ind_map[word]=word_ind
        word_ind=word_ind+1
    print(len(word_ind_map.keys()))
    word_numer=len(word_ind_map.keys())
    sample_number=len(lines)-2
    features=np.zeros((sample_number,word_numer),dtype=float)
    node_list=[]
    label_list=[]

    line_number=0
    for line in lines[2:]:
        """
        12187484	label=1	w-rat=0.09393489570187145	w-common=0.028698458467273157	summary=w-rat,w-studi
        """
        slices=line.split('\t')
        node_name=slices[0]
        node_label=slices[1].split('=')[1]
        node_list.append(node_name)
        label_list.append(node_label)
        for word_value in slices[2:-1]:
            [word,value]=word_value.split('=')
            features[line_number,word_ind_map[word]]=value
        line_number=line_number+1
    left=np.asarray(node_list).reshape(-1,1)
    right=np.asarray(label_list).reshape(-1,1)
    print(left.shape)
    print(right.shape)
    content=np.concatenate((left,features,right),axis=1)
    np.savetxt('./pubmed.content', content, fmt='%s', delimiter='\t')














# raw_edges=np.loadtxt('./Pubmed-Diabetes.DIRECTED.cites.tab',delimiter='\t',dtype=str)
#
# print(raw_edges)
# print(raw_edges.shape)
# raw_edges=raw_edges[:,[1,3]]
# # raw_edges=np.concatenate((raw_edges[:,1],raw_edges[:,3]),axis=1)
# print(raw_edges)
# print(raw_edges.shape)
# with open('./pubmed.cites','w') as fout:
#     for line in raw_edges:
#         src=line[0].split(':')[1]
#         des=line[1].split(':')[1]
#         fout.write(src+'\t'+des+'\n')





