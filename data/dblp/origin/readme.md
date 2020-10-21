DBLP is a bibliographic network in computer science
collected from four research areas: database, data
mining, machine learning, and information retrieval. In
the dataset, 4057 authors, 20 venues and 100 papers are
labeled with one of the four research areas.

Yizhou Sun, Jiawei Han, Xifeng Yan, Philip S Yu, and Tianyi Wu. 2011. Pathsim:
Meta path-based top-k similarity search in heterogeneous information networks.
Proceedings of the VLDB Endowment 4, 11 (2011), 992–1003.



这里有四种节点
t：term单词，只与p相连
a：author作者，有标签
p：论文，有标签
c：会议，有标签

边只有p-a，p-t，p-c这三种

处理措施
生成dblp.cites
移除p-t这一类型
生成dblp.content,将邻接矩阵作为属性，维度在4177，可以接受
