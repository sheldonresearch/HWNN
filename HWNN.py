#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@project:HWNN
@author:xiangguosun 
@contact:sunxiangguodut@qq.com
@website:http://blog.csdn.net/github_36326955
@file: HWNN.py 
@platform: macOS High Sierra 10.13.1 Pycharm pro 2017.1 
@time: 2019/10/16 
"""
import os
import torch
import torch.nn.functional as F
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run HWNN.")

    parser.add_argument("--epochs",
                        type=int,
                        default=300,  # 600
                        help="Number of training epochs. Default is 300.")

    parser.add_argument("--filters",
                        type=int,
                        default=128,
                        help="Filters (neurons) in convolution. Default is 128.")

    parser.add_argument("--test-size",
                        type=float,
                        default=0.2,
                        help="Ratio of training samples. Default is 0.8")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.01,
                        help="Dropout probability. Default is 0.01")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for sklearn pre-training. Default is 42.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
                        help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=0.0001,
                        help="Adam weight decay. Default is 0.0001.")

    return parser.parse_args()


class HWNNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ncount, device, K1=2, K2=2, approx=False, data=None):
        super(HWNNLayer, self).__init__()
        self.data = data
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncount = ncount
        self.device = device
        self.K1 = K1
        self.K2 = K2
        self.approx = approx
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.diagonal_weight_filter = torch.nn.Parameter(torch.Tensor(self.ncount))
        self.par = torch.nn.Parameter(torch.Tensor(self.K1 + self.K2))
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.99, 1.01)
        torch.nn.init.uniform_(self.par, 0, 0.99)

    def forward(self, features, snap_index, data):
        diagonal_weight_filter = torch.diag(self.diagonal_weight_filter).to(self.device)
        features = features.to(self.device)
        # Theta=self.data.Theta.to(self.device)
        Theta = data.hypergraph_snapshot[snap_index]["Theta"].to(self.device)
        Theta_t = torch.transpose(Theta, 0, 1)

        if self.approx:
            poly = self.par[0] * torch.eye(self.ncount).to(self.device)
            Theta_mul = torch.eye(self.ncount).to(self.device)
            for ind in range(1, self.K1):
                Theta_mul = Theta_mul @ Theta
                poly = poly + self.par[ind] * Theta_mul

            poly_t = self.par[self.K1] * torch.eye(self.ncount).to(self.device)
            Theta_mul = torch.eye(self.ncount).to(self.device)
            for ind in range(self.K1 + 1, self.K1 + self.K2):
                Theta_mul = Theta_mul @ Theta_t  # 这里也可以使用Theta_transpose
                poly_t = poly_t + self.par[ind] * Theta_mul

            # poly=self.par[0]*torch.eye(self.ncount).to(self.device)+self.par[1]*Theta+self.par[2]*Theta@Theta
            # poly_t = self.par[3] * torch.eye(self.ncount).to(self.device) + self.par[4] * Theta_t + self.par[5] * Theta_t @ Theta_t
            # poly_t = self.par[3] * torch.eye(self.ncount).to(self.device) + self.par[4] * Theta + self.par[
            #     5] * Theta @ Theta
            local_fea_1 = poly @ diagonal_weight_filter @ poly_t @ features @ self.weight_matrix
        else:
            print("wavelets!")
            wavelets = self.data.hypergraph_snapshot[snap_index]["wavelets"].to(self.device)
            wavelets_inverse = self.data.hypergraph_snapshot[snap_index]["wavelets_inv"].to(self.device)
            local_fea_1 = wavelets @ diagonal_weight_filter @ wavelets_inverse @ features @ self.weight_matrix

        localized_features = local_fea_1
        return localized_features


class HWNN(torch.nn.Module):
    def __init__(self, args, ncount, feature_number, class_number, device, data):
        super(HWNN, self).__init__()
        self.args = args
        # self.features=features
        self.ncount = ncount
        self.feature_number = feature_number

        self.class_number = class_number
        self.device = device
        self.data = data

        self.hyper_snapshot_num = len(self.data.hypergraph_snapshot)
        print("there are {} hypergraphs".format(self.hyper_snapshot_num))


        self.setup_layers()

        self.par = torch.nn.Parameter(torch.Tensor(self.hyper_snapshot_num))
        torch.nn.init.uniform_(self.par, 0, 0.99)  # 1.0)

    def setup_layers(self):
        self.convolution_1 = HWNNLayer(self.feature_number,
                                       self.args.filters,
                                       self.ncount,
                                       self.device,
                                       K1=3,
                                       K2=3,
                                       approx=True,
                                       data=self.data)

        self.convolution_2 = HWNNLayer(self.args.filters,
                                       self.class_number,
                                       self.ncount,
                                       self.device,
                                       K1=3,
                                       K2=3,
                                       approx=True,
                                       data=self.data)

    def forward(self, features):
        features = features.to(self.device)
        channel_feature = []
        for snap_index in range(self.hyper_snapshot_num):
            deep_features_1 = F.relu(self.convolution_1(features,
                                                        snap_index,
                                                        self.data))
            deep_features_1 = F.dropout(deep_features_1, self.args.dropout)
            deep_features_2 = self.convolution_2(deep_features_1,
                                                 snap_index,
                                                 self.data)
            deep_features_2 = F.log_softmax(deep_features_2, dim=1)  # 把这里换成relu会怎么样呢？
            channel_feature.append(deep_features_2)

        deep_features_3 = torch.zeros_like(channel_feature[0])
        for ind in range(self.hyper_snapshot_num):
            deep_features_3 = deep_features_3 + self.par[ind] * channel_feature[ind]

        return deep_features_3


class HWNNTrainer(object):
    def __init__(self, args, features, target, data):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.args = args
        self.data = data

        self.features = features
        self.ncount = self.features.size()[0]
        self.feature_number = self.features.size()[1]

        self.target = target.to(self.device)
        self.class_number = self.data.class_num

        self.setup_model()
        self.train_test_split()

    def setup_model(self):
        self.model = HWNN(self.args,
                          self.ncount,
                          self.feature_number,
                          self.class_number,
                          self.device, self.data).to(self.device)

    def train_test_split(self):
        nodes = [x for x in range(self.ncount)]
        train_nodes, test_nodes = train_test_split(nodes, test_size=self.args.test_size, random_state=self.args.seed)
        self.train_nodes = torch.LongTensor(train_nodes)
        self.test_nodes = torch.LongTensor(test_nodes)

    def fit(self):
        print("HWNN Training.\n")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        self.model.train()
        self.best_accuracy = 0.0
        self.best_micro_f1 = 0.0
        self.best_macro_f1 = 0.0
        self.best_precision = 0.0
        self.best_recall = 0.0
        # writer = SummaryWriter()

        for epoch in range(self.args.epochs):
            self.time = time.time()
            self.optimizer.zero_grad()

            prediction = self.model(self.features)

            loss_train = torch.nn.functional.cross_entropy(prediction[self.train_nodes],
                                                           self.target[self.train_nodes])

            loss_test = torch.nn.functional.cross_entropy(prediction[self.test_nodes],
                                                          self.target[self.test_nodes])

            loss_train.backward()
            self.optimizer.step()

            _, prediction = self.model(self.features).max(dim=1)

            """
            self.target[self.test_nodes].cpu()
            tensor([2, 2, 1,  ..., 5, 2, 0])
            
            prediction[self.test_nodes].cpu()
            tensor([6, 0, 2,  ..., 6, 2, 0])
            """

            # # accuracy
            correct_test = prediction[self.test_nodes].eq(self.target[self.test_nodes]).sum().item()
            accuracy_test = correct_test / int(self.ncount * self.args.test_size)
            correct_train = prediction[self.train_nodes].eq(self.target[self.train_nodes]).sum().item()
            accuracy_train = correct_train / int(self.ncount * (1 - self.args.test_size))

            # micro F1
            """
            >>> y_true = [0, 1, 2, 0, 1, 2]
            >>> y_pred = [0, 2, 1, 0, 0, 1]
            """

            micro_f1 = metrics.f1_score(self.target[self.test_nodes].cpu(),
                                        prediction[self.test_nodes].cpu(),
                                        average='micro')

            # macro F1
            macro_f1 = metrics.f1_score(self.target[self.test_nodes].cpu(),
                                        prediction[self.test_nodes].cpu(),
                                        average='macro')

            # precision
            precision = metrics.precision_score(self.target[self.test_nodes].cpu(),
                                                prediction[self.test_nodes].cpu(),
                                                average='macro')
            # recall
            recall = metrics.recall_score(self.target[self.test_nodes].cpu(),
                                          prediction[self.test_nodes].cpu(),
                                          average='macro')

            if self.best_accuracy < accuracy_test:
                self.best_accuracy = accuracy_test

            if self.best_micro_f1 < micro_f1:
                self.best_micro_f1 = micro_f1
            if self.best_macro_f1 < macro_f1:
                self.best_macro_f1 = macro_f1
            if self.best_precision < precision:
                self.best_precision = precision
            if self.best_recall < recall:
                self.best_recall = recall

            print("epo:{}/{}|"
                  "train_los:{:.4f}|"
                  "test_los:{:.4f}|"
                  "train_acc:{:.4f}|"
                  "test_acc:{:.4f}|"
                  "best_acc:{:.4f}|"
                  "best_micro_f1:{:.4f}|"
                  "best_macro_f1:{:.4f}|"
                  "best_precision:{:.4f}|"
                  "best_recall:{:.4f}".format(epoch, self.args.epochs,
                                              loss_train,
                                              loss_test,
                                              accuracy_train,
                                              accuracy_test,
                                              self.best_accuracy,
                                              self.best_micro_f1,
                                              self.best_macro_f1,
                                              self.best_precision,
                                              self.best_recall))


if __name__ == "__main__":
    from data import Data

    args = parameter_parser()
    data = Data()
    # data.load(data_path='./data/spammer/',data_name='spammer')
    data.load(data_path='./data/cora/', data_name='cora')

    target = data.nodes_labels_sequence.type(torch.LongTensor)
    features = data.X_0.type(torch.FloatTensor)

    trainer = HWNNTrainer(args, features, target, data)
    trainer.fit()
