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


dblp.cites
src\tdes\ttype
p9896	a6716   0
p       c       1

dblp.content
p 临街矩阵 标签
a   临街矩阵 标签

请注意：content里的内容不能直接从原始文件中拷贝，
因为有些节点可能是孤立的，因此要从原始的label文件里抽取
抽取的依据是他们在边文件中是否存在
"""
import numpy as np
from collections import defaultdict
import torch


dblp_labels=np.loadtxt('./origin/dblp_labels.txt',dtype=str)
all_nodes_names=list(dblp_labels[:,0])


dblp=np.loadtxt('./origin/dblp.txt',dtype=str)

nodes_set=set()

with open('./dblp.cites','w') as fout:
    for row in dblp:
        if row[0] in all_nodes_names and row[1] in all_nodes_names:
            nodes_set.add(row[0])
            nodes_set.add(row[1])
            if 'a' in row[1]:
                edge_type='0'
            elif 'c' in row[1]:
                edge_type='1'
            else:
                break
            fout.write(row[0]+'\t'+row[1]+'\t'+edge_type+'\n')



dblp=np.loadtxt('./dblp.cites',dtype=str)
# print(dblp)
# print(dblp_labels)









filter_labels=[]
for row in dblp_labels:
    if row[0] in nodes_set:
        filter_labels.append(row)


dblp_labels=np.asarray(filter_labels)
all_nodes_names=list(dblp_labels[:,0])
graph=defaultdict(list)
A=np.zeros((dblp_labels.shape[0],dblp_labels.shape[0]),dtype=np.int)


nodes_names_map=defaultdict(int)
i=0
for row in dblp_labels:
    nodes_names_map[row[0]]=i
    i=i+1
# print(nodes_names_map)

for edge in dblp:
    # print(edge,nodes_names_map[edge[0]],nodes_names_map[edge[1]])
    A[nodes_names_map[edge[0]],nodes_names_map[edge[1]]]=1
    A[nodes_names_map[edge[1]], nodes_names_map[edge[0]]] = 1
# A=A.astype(str)
content=np.concatenate((dblp_labels[:,0].reshape(-1,1),A,dblp_labels[:,1].reshape(-1,1)),axis=1)
print(content)
np.savetxt('./dblp.content',content,fmt='%s',delimiter='\t')
# for edge in dblp:
#     graph[edge[0]].append(edge[1])
#     graph[edge[1]].append(edge[0])
#
# print(graph)
# for item in graph.items():
#     if item[0] in item[1]:
#         item[1].remove(item[0])
#     graph[item[0]] = np.unique(item[1])
#     for node in graph[item[0]]:
#         print(nodes_names_map[item[0]],nodes_names_map[node])
#         A[nodes_names_map[item[0]], nodes_names_map[node]] = 1
#
# print(A)




