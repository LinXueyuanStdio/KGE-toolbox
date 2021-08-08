from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import click

from toolbox.Embed import get_vec2
from toolbox.Progbar import Progbar
from toolbox.Store import load_checkpoint, save_checkpoint, save_entity_embedding_list
from toolbox.dataloader import TrainDataset, AlignDataset, BidirectionalOneShotIterator
from toolbox.DataSchema import DBP15kData, DBP15kCachePath, append_align_triple, read_cache, cache_data
from toolbox.DatasetSchema import DBP15k
from toolbox.OutputSchema import OutputSchema
from toolbox.Evaluate import evaluate_entity_alignment, get_score, pretty_print

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)


# region model
class Kert(nn.Module):
    def __init__(self,
                 train_seeds,
                 nentity, nrelation, nvalue,
                 hidden_dim, gamma):
        super(Kert, self).__init__()
        # self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.dim = nn.Parameter(
            torch.Tensor([25]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        self.value_dim = hidden_dim

        # region 知识图谱的嵌入：实体、属性、属性值
        entity_weight = torch.zeros(nentity, self.entity_dim)
        nn.init.uniform_(
            tensor=entity_weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        for left_entity, right_entity in train_seeds:
            entity_weight[left_entity] = entity_weight[right_entity]
        self.entity_embedding = nn.Parameter(entity_weight)
        # nn.init.normal_(self.entity_embedding)

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        # nn.init.normal_(self.relation_embedding)
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.relation_embedding2 = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        # nn.init.normal_(self.relation_embedding)
        nn.init.uniform_(
            tensor=self.relation_embedding2,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.relation_embedding3 = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        # nn.init.normal_(self.relation_embedding)
        nn.init.uniform_(
            tensor=self.relation_embedding3,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.relation_embedding4 = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        # nn.init.normal_(self.relation_embedding)
        nn.init.uniform_(
            tensor=self.relation_embedding4,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        # self.relation_embedding5 = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        # # nn.init.normal_(self.relation_embedding5)
        # nn.init.uniform_(
        #     tensor=self.relation_embedding5,
        #     a=-self.embedding_range.item(),
        #     b=self.embedding_range.item()
        # )
        # self.relation_embedding6 = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        # # nn.init.normal_(self.relation_embedding6)
        # nn.init.uniform_(
        #     tensor=self.relation_embedding6,
        #     a=-self.embedding_range.item(),
        #     b=self.embedding_range.item()
        # )
        #
        self.K = nn.Parameter(torch.zeros(8, 25))
        self.V = nn.Parameter(torch.zeros(8, 25))
        nn.init.normal_(self.K)
        nn.init.normal_(self.V)
        self.K2 = nn.Parameter(torch.zeros(8, 25))
        self.V2 = nn.Parameter(torch.zeros(8, 25))
        nn.init.normal_(self.K2)
        nn.init.normal_(self.V2)
        self.K3 = nn.Parameter(torch.zeros(8, 25))
        self.V3 = nn.Parameter(torch.zeros(8, 25))
        nn.init.normal_(self.K3)
        nn.init.normal_(self.V3)
        self.K4 = nn.Parameter(torch.zeros(8, 25))
        self.V4 = nn.Parameter(torch.zeros(8, 25))
        nn.init.normal_(self.K4)
        nn.init.normal_(self.V4)
        # self.K5 = nn.Parameter(torch.zeros(8, 25))
        # self.V5 = nn.Parameter(torch.zeros(8, 25))
        # nn.init.normal_(self.K5)
        # nn.init.normal_(self.V5)
        # self.K6 = nn.Parameter(torch.zeros(8, 25))
        # self.V6 = nn.Parameter(torch.zeros(8, 25))
        # nn.init.normal_(self.K6)
        # nn.init.normal_(self.V6)

        # endregion

    def forward(self, sample, mode='single'):
        # region align 对齐loss 使用GCN-Align的对齐模块
        if mode == "align-single":
            batch_size, negative_sample_size = sample.size(0), 1
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
            return self.align_loss(head, tail, mode)
        elif mode == 'align-head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)
            return self.align_loss(head, tail, mode)
        elif mode == 'align-tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            return self.align_loss(head, tail, mode)
        # endregion

        # 以下是 Kert
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
            relation2 = torch.index_select(
                self.relation_embedding2,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
            relation3 = torch.index_select(
                self.relation_embedding3,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
            relation4 = torch.index_select(
                self.relation_embedding4,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
            # relation5 = torch.index_select(
            #     self.relation_embedding5,
            #     dim=0,
            #     index=sample[:, 1]
            # ).unsqueeze(1)
            # relation6 = torch.index_select(
            #     self.relation_embedding6,
            #     dim=0,
            #     index=sample[:, 1]
            # ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            # head_part : batch_size x sample_size
            # tail_part : batch_size x 3
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)
            relation2 = torch.index_select(
                self.relation_embedding2,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)
            relation3 = torch.index_select(
                self.relation_embedding3,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)
            relation4 = torch.index_select(
                self.relation_embedding4,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)
            # relation5 = torch.index_select(
            #     self.relation_embedding5,
            #     dim=0,
            #     index=tail_part[:, 1]
            # ).unsqueeze(1)
            # relation6 = torch.index_select(
            #     self.relation_embedding6,
            #     dim=0,
            #     index=tail_part[:, 1]
            # ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':

            head_part, tail_part = sample
            # head_part : batch_size x 3
            # tail_part : batch_size x sample_size
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            relation2 = torch.index_select(
                self.relation_embedding2,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            relation3 = torch.index_select(
                self.relation_embedding3,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            relation4 = torch.index_select(
                self.relation_embedding4,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            # relation5 = torch.index_select(
            #     self.relation_embedding5,
            #     dim=0,
            #     index=head_part[:, 1]
            # ).unsqueeze(1)
            # relation6 = torch.index_select(
            #     self.relation_embedding6,
            #     dim=0,
            #     index=head_part[:, 1]
            # ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        # score = self.Attn10(head, relation, relation2, relation3, relation4, relation5, relation6, tail, mode)
        score = self.Attn7(head, relation, relation2, relation3, relation4, tail, mode)
        # score = self.Attn2(head, relation, relation2, tail, mode)
        # score = self.RotatE(head, relation, tail, mode)
        return score

    def align_loss(self, head, tail, mode):
        # print(mode, head.size(), tail.size())
        score = head - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        # score = torch.norm(score, p=1, dim=2)
        return score

    def attention(self, Q, K, V):
        return Q.matmul(K.transpose(2, 3)).div(self.dim).softmax(3).matmul(V)

    def attention2(self, Q, K_T, V):
        return Q.matmul(K_T).div(self.dim).softmax(3).matmul(V)

    def self_attention(self, X, Wq, Wk, Wv):
        Q = X.matmul(Wq)
        K = X.matmul(Wk)
        V = X.matmul(Wv)
        return self.attention(Q, K, V)

    def relation_self_attention(self, X, Wq, Wk, Wv, relK, relV, Wq2, Wk2, Wv2):
        attnX = self.self_attention(X, Wq, Wk, Wv) + X
        attnX = attnX.tanh()
        attnX = self.attention(attnX, relK, relV) + attnX
        attnX = attnX.tanh()
        attnX = self.self_attention(attnX, Wq2, Wk2, Wv2) + attnX
        attnX = attnX.tanh()
        return attnX

    def relation_attention(self, X, K, V, relK, relV, K2, V2):
        attnX = self.attention2(X, K.transpose(0, 1), V) + X
        attnX = attnX.tanh()
        attnX = self.attention(attnX, relK, relV) + attnX
        attnX = attnX.tanh()
        attnX = self.attention2(attnX, K2.transpose(0, 1), V2) + attnX
        attnX = attnX.tanh()
        return attnX

    def Attn10(self, head, relation, relation2, relation3, relation4, relation5, relation6, tail, mode):
        """
        head: batch_size x 1 x embedding_size (head-batch: batch_size x sample_size x embedding_size)
        relation: batch_size x 1 x embedding_size
        tail: batch_size x 1 x embedding_size (tail-batch: batch_size x sample_size x embedding_size)

        attention = V * softmax(K^T * Q / sqrt(d))
        Q = head, K,V = relation, attention = tail

        head - (ShareK,ShareV)   - (RelK,RelV)   - (ShareK2,ShareV2) --->|
                                                                         |----> score
        tail - (ShareK3,ShareV3) - (RelK2,RelV2) - (ShareK4,ShareV4) --->|

        效果不好
        """
        if mode == 'head-batch':
            batch_size, sample_size, embedding_size = head.size()
            Q = head.view(batch_size, sample_size, 8, 25)
            T = tail.view(batch_size, 1, 8, 25)
        else:
            batch_size, sample_size, embedding_size = tail.size()
            Q = head.view(batch_size, 1, 8, 25)
            T = tail.view(batch_size, sample_size, 8, 25)
        K = relation.view(batch_size, 1, 8, 25)
        V = relation2.view(batch_size, 1, 8, 25)
        K2 = relation3.view(batch_size, 1, 8, 25)
        V2 = relation4.view(batch_size, 1, 8, 25)
        K3 = relation5.view(batch_size, 1, 8, 25)
        V3 = relation6.view(batch_size, 1, 8, 25)

        AttnQ = Q + self.relation_attention(Q, self.K, self.V, K, V, self.K2, self.V2)
        AttnQ.tanh()
        AttnT = T + self.relation_attention(T, self.K3, self.V3, K2, V2, self.K4, self.V4)
        AttnT.tanh()
        score = AttnQ + self.relation_attention(AttnT, self.K5, self.V5, K3, V3, self.K6, self.V6)
        score.tanh()
        score = score.view(batch_size, sample_size, embedding_size)
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def Attn9(self, head, relation, relation2, relation3, relation4, tail, mode):
        """
        head: batch_size x 1 x embedding_size (head-batch: batch_size x sample_size x embedding_size)
        relation: batch_size x 1 x embedding_size
        tail: batch_size x 1 x embedding_size (tail-batch: batch_size x sample_size x embedding_size)

        attention = V * softmax(K^T * Q / sqrt(d))
        Q = head, K,V = relation, attention = tail

        head - (ShareK,ShareV)   - (RelK,RelV)   - (ShareK2,ShareV2) --->|
                                                                         |----> score
        tail - (ShareK3,ShareV3) - (RelK2,RelV2) - (ShareK4,ShareV4) --->|

        self Attention
        """
        if mode == 'head-batch':
            batch_size, sample_size, embedding_size = head.size()
            Q = head.view(batch_size, sample_size, 8, 25)
            T = tail.view(batch_size, 1, 8, 25)
        else:
            batch_size, sample_size, embedding_size = tail.size()
            Q = head.view(batch_size, 1, 8, 25)
            T = tail.view(batch_size, sample_size, 8, 25)
        K = relation.view(batch_size, 1, 8, 25)
        V = relation2.view(batch_size, 1, 8, 25)
        K2 = relation3.view(batch_size, 1, 8, 25)
        V2 = relation4.view(batch_size, 1, 8, 25)

        AttnQ = Q + self.relation_self_attention(Q, self.Wq, self.Wk, self.Wv, K, V, self.Wq2, self.Wk2, self.Wv2)
        AttnT = T + self.relation_self_attention(T, self.Wq3, self.Wk3, self.Wv3, K2, V2, self.Wq4, self.Wk4, self.Wv4)

        score = AttnT - AttnQ
        score = score.view(batch_size, sample_size, embedding_size)
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def Attn8(self, head, relation, relation2, relation3, relation4, tail, mode):
        """
        head: batch_size x 1 x embedding_size (head-batch: batch_size x sample_size x embedding_size)
        relation: batch_size x 1 x embedding_size
        tail: batch_size x 1 x embedding_size (tail-batch: batch_size x sample_size x embedding_size)

        attention = V * softmax(K^T * Q / sqrt(d))
        Q = head, K,V = relation, attention = tail

        head - (ShareK,ShareV)     (RelK,RelV)     (ShareK2,ShareV2) --->|
                                 x               x                       |----> score
        tail - (ShareK3,ShareV3)   (RelK2,RelV2)   (ShareK4,ShareV4) --->|

        效果极差
        hits1 < 1
        """
        if mode == 'head-batch':
            batch_size, sample_size, embedding_size = head.size()
            Q = head.view(batch_size, sample_size, 8, 25)
            T = tail.view(batch_size, 1, 8, 25)
        else:
            batch_size, sample_size, embedding_size = tail.size()
            Q = head.view(batch_size, 1, 8, 25)
            T = tail.view(batch_size, sample_size, 8, 25)
        K = relation.view(batch_size, 1, 8, 25)
        V = relation2.view(batch_size, 1, 8, 25)
        K2 = relation3.view(batch_size, 1, 8, 25)
        V2 = relation4.view(batch_size, 1, 8, 25)
        SK = self.share_relation.view(8, 25)
        SV = self.share_relation2.view(8, 25)
        SK2 = self.share_relation3.view(8, 25)
        SV2 = self.share_relation4.view(8, 25)
        SK3 = self.share_relation5.view(8, 25)
        SV3 = self.share_relation6.view(8, 25)
        SK4 = self.share_relation7.view(8, 25)
        SV4 = self.share_relation8.view(8, 25)

        AttnQ = Q.matmul(SK.transpose(0, 1)).div(self.dim).softmax(3).matmul(SV) + Q
        AttnQ = AttnQ.tanh()
        AttnT = T.matmul(SK3.transpose(0, 1)).div(self.dim).softmax(3).matmul(SV3) + T
        AttnT = AttnT.tanh()

        AttnQ = AttnQ.matmul(K.transpose(2, 3)).div(self.dim).softmax(3).matmul(V) + AttnT
        AttnQ = AttnQ.tanh()
        AttnT = AttnT.matmul(K2.transpose(2, 3)).div(self.dim).softmax(3).matmul(V2) + AttnQ
        AttnT = AttnT.tanh()

        AttnQ = AttnQ.matmul(SK2.transpose(0, 1)).div(self.dim).softmax(3).matmul(SV2) + AttnQ
        AttnQ = AttnQ.tanh()
        AttnT = AttnT.matmul(SK4.transpose(0, 1)).div(self.dim).softmax(3).matmul(SV4) + AttnT
        AttnT = AttnT.tanh()

        score = (AttnT + AttnQ).sigmoid()
        score = score.view(batch_size, sample_size, embedding_size)
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def Attn7(self, head, relation, relation2, relation3, relation4, tail, mode):
        """
        head: batch_size x 1 x embedding_size (head-batch: batch_size x sample_size x embedding_size)
        relation: batch_size x 1 x embedding_size
        tail: batch_size x 1 x embedding_size (tail-batch: batch_size x sample_size x embedding_size)

        attention = V * softmax(K^T * Q / sqrt(d))
        Q = head, K,V = relation, attention = tail

        head - (ShareK,ShareV)   - (RelK,RelV)   - (ShareK2,ShareV2) --->|
                                                                         |----> score
        tail - (ShareK3,ShareV3) - (RelK2,RelV2) - (ShareK4,ShareV4) --->|

        ---
        self.share_relation = nn.Parameter(torch.zeros(self.relation_dim))
        nn.init.normal_(self.share_relation)
        self.share_relation2 = nn.Parameter(torch.zeros(self.relation_dim))
        nn.init.normal_(self.share_relation2)
        self.share_relation3 = nn.Parameter(torch.zeros(self.relation_dim))
        nn.init.normal_(self.share_relation3)
        self.share_relation4 = nn.Parameter(torch.zeros(self.relation_dim))
        nn.init.normal_(self.share_relation4)

        self.share_relation5 = nn.Parameter(torch.zeros(self.relation_dim))
        nn.init.normal_(self.share_relation5)
        self.share_relation6 = nn.Parameter(torch.zeros(self.relation_dim))
        nn.init.normal_(self.share_relation6)
        self.share_relation7 = nn.Parameter(torch.zeros(self.relation_dim))
        nn.init.normal_(self.share_relation7)
        self.share_relation8 = nn.Parameter(torch.zeros(self.relation_dim))
        nn.init.normal_(self.share_relation8)
        ---

        hits1 in [40, 45)
        """
        if mode == 'head-batch':
            batch_size, sample_size, embedding_size = head.size()
            Q = head.view(batch_size, sample_size, 8, 25)
            T = tail.view(batch_size, 1, 8, 25)
        else:
            batch_size, sample_size, embedding_size = tail.size()
            Q = head.view(batch_size, 1, 8, 25)
            T = tail.view(batch_size, sample_size, 8, 25)
        K = relation.view(batch_size, 1, 8, 25)
        V = relation2.view(batch_size, 1, 8, 25)
        K2 = relation3.view(batch_size, 1, 8, 25)
        V2 = relation4.view(batch_size, 1, 8, 25)
        SK = self.K
        SV = self.V
        SK2 = self.K2
        SV2 = self.V2
        SK3 = self.K3
        SV3 = self.V3
        SK4 = self.K4
        SV4 = self.V4

        AttnQ = Q.matmul(SK.transpose(0, 1)).div(self.dim).softmax(3).matmul(SV) + Q
        AttnQ = AttnQ.tanh()
        AttnQ = AttnQ.matmul(K.transpose(2, 3)).div(self.dim).softmax(3).matmul(V) + AttnQ
        AttnQ = AttnQ.tanh()
        AttnQ = AttnQ.matmul(SK2.transpose(0, 1)).div(self.dim).softmax(3).matmul(SV2) + AttnQ
        AttnQ = AttnQ.tanh()

        AttnT = T.matmul(SK3.transpose(0, 1)).div(self.dim).softmax(3).matmul(SV3) + T
        AttnT = AttnT.tanh()
        AttnT = AttnT.matmul(K2.transpose(2, 3)).div(self.dim).softmax(3).matmul(V2) + AttnT
        AttnT = AttnT.tanh()
        AttnT = AttnT.matmul(SK4.transpose(0, 1)).div(self.dim).softmax(3).matmul(SV4) + AttnT
        AttnT = AttnT.tanh()

        score = AttnT - AttnQ
        score = score.view(batch_size, sample_size, embedding_size)
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def Attn6(self, head, relation, relation2, relation3, relation4, tail, mode):
        """
        head: batch_size x 1 x embedding_size (head-batch: batch_size x sample_size x embedding_size)
        relation: batch_size x 1 x embedding_size
        tail: batch_size x 1 x embedding_size (tail-batch: batch_size x sample_size x embedding_size)

        attention = V * softmax(K^T * Q / sqrt(d))
        Q = head, K,V = relation, attention = tail

        head - (ShareK,ShareV) - (RelK,RelV) - (RelK2,RelV2) - (ShareK2,ShareV2) -> tail

        hits1 = 40
        """
        SK = self.share_relation.view(8, 25)
        SV = self.share_relation2.view(8, 25)
        SK2 = self.share_relation3.view(8, 25)
        SV2 = self.share_relation4.view(8, 25)
        if mode == 'head-batch':
            batch_size, sample_size, embedding_size = head.size()
            Q = head.view(batch_size, sample_size, 8, 25)
            K = relation.view(batch_size, 1, 8, 25)
            V = relation2.view(batch_size, 1, 8, 25)
            K2 = relation3.view(batch_size, 1, 8, 25)
            V2 = relation4.view(batch_size, 1, 8, 25)
            Attn = tail.view(batch_size, 1, 8, 25)
            AttnQ = Q.matmul(SK.transpose(0, 1)).div(self.dim).softmax(3).matmul(SV) + Q
            AttnQ = AttnQ.tanh()
            AttnQ = AttnQ.matmul(K.transpose(2, 3)).div(self.dim).softmax(3).matmul(V) + AttnQ
            AttnQ = AttnQ.tanh()
            AttnQ = AttnQ.matmul(K2.transpose(2, 3)).div(self.dim).softmax(3).matmul(V2) + AttnQ
            AttnQ = AttnQ.tanh()
            AttnQ = AttnQ.matmul(SK2.transpose(0, 1)).div(self.dim).softmax(3).matmul(SV2) + AttnQ
            AttnQ = AttnQ.tanh()
            score = AttnQ - Attn
            score = score.view(batch_size, sample_size, embedding_size)
            # score = head + (relation - tail)
        else:
            batch_size, sample_size, embedding_size = tail.size()
            Q = head.view(batch_size, 1, 8, 25)
            K = relation.view(batch_size, 1, 8, 25)
            V = relation2.view(batch_size, 1, 8, 25)
            K2 = relation3.view(batch_size, 1, 8, 25)
            V2 = relation4.view(batch_size, 1, 8, 25)
            Attn = tail.view(batch_size, sample_size, 8, 25)
            AttnQ = Q.matmul(SK.transpose(0, 1)).div(self.dim).softmax(3).matmul(SV) + Q
            AttnQ = AttnQ.tanh()
            AttnQ = AttnQ.matmul(K.transpose(2, 3)).div(self.dim).softmax(3).matmul(V) + AttnQ
            AttnQ = AttnQ.tanh()
            AttnQ = AttnQ.matmul(K2.transpose(2, 3)).div(self.dim).softmax(3).matmul(V2) + AttnQ
            AttnQ = AttnQ.tanh()
            AttnQ = AttnQ.matmul(SK2.transpose(0, 1)).div(self.dim).softmax(3).matmul(SV2) + AttnQ
            AttnQ = AttnQ.tanh()
            score = AttnQ - Attn
            score = score.view(batch_size, sample_size, embedding_size)
            # score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def Attn5(self, head, relation, relation2, relation3, relation4, tail, mode):
        """
        head: batch_size x 1 x embedding_size (head-batch: batch_size x sample_size x embedding_size)
        relation: batch_size x 1 x embedding_size
        tail: batch_size x 1 x embedding_size (tail-batch: batch_size x sample_size x embedding_size)

        attention = V * softmax(K^T * Q / sqrt(d))
        Q = head, K,V = relation, attention = tail

        head - (RelK,RelV) - (ShareK,ShareV) - (ShareK2,ShareV2) - (RelK2,RelV2) -> tail

        hits1 = 40
        """
        SK = self.share_relation.view(8, 25)
        SV = self.share_relation2.view(8, 25)
        SK2 = self.share_relation3.view(8, 25)
        SV2 = self.share_relation4.view(8, 25)
        if mode == 'head-batch':
            batch_size, sample_size, embedding_size = head.size()
            Q = head.view(batch_size, sample_size, 8, 25)
            K = relation.view(batch_size, 1, 8, 25)
            V = relation2.view(batch_size, 1, 8, 25)
            K2 = relation3.view(batch_size, 1, 8, 25)
            V2 = relation4.view(batch_size, 1, 8, 25)
            Attn = tail.view(batch_size, 1, 8, 25)
            AttnQ = Q.matmul(K.transpose(2, 3)).div(self.dim).softmax(3).matmul(V) + Q
            AttnQ = AttnQ.tanh()
            AttnQ = AttnQ.matmul(SK.transpose(0, 1)).div(self.dim).softmax(3).matmul(SV) + AttnQ
            AttnQ = AttnQ.tanh()
            AttnQ = AttnQ.matmul(SK2.transpose(0, 1)).div(self.dim).softmax(3).matmul(SV2) + AttnQ
            AttnQ = AttnQ.tanh()
            AttnQ = AttnQ.matmul(K2.transpose(2, 3)).div(self.dim).softmax(3).matmul(V2) + AttnQ
            AttnQ = AttnQ.tanh()
            score = AttnQ - Attn
            score = score.view(batch_size, sample_size, embedding_size)
            # score = head + (relation - tail)
        else:
            batch_size, sample_size, embedding_size = tail.size()
            Q = head.view(batch_size, 1, 8, 25)
            K = relation.view(batch_size, 1, 8, 25)
            V = relation2.view(batch_size, 1, 8, 25)
            K2 = relation3.view(batch_size, 1, 8, 25)
            V2 = relation4.view(batch_size, 1, 8, 25)
            Attn = tail.view(batch_size, sample_size, 8, 25)
            AttnQ = Q.matmul(K.transpose(2, 3)).div(self.dim).softmax(3).matmul(V) + Q
            AttnQ = AttnQ.tanh()
            AttnQ = AttnQ.matmul(SK.transpose(0, 1)).div(self.dim).softmax(3).matmul(SV) + AttnQ
            AttnQ = AttnQ.tanh()
            AttnQ = AttnQ.matmul(SK2.transpose(0, 1)).div(self.dim).softmax(3).matmul(SV2) + AttnQ
            AttnQ = AttnQ.tanh()
            AttnQ = AttnQ.matmul(K2.transpose(2, 3)).div(self.dim).softmax(3).matmul(V2) + AttnQ
            AttnQ = AttnQ.tanh()
            score = AttnQ - Attn
            score = score.view(batch_size, sample_size, embedding_size)
            # score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def Attn4(self, head, relation, relation2, relation3, relation4, tail, mode):
        """
        head: batch_size x 1 x embedding_size (head-batch: batch_size x sample_size x embedding_size)
        relation: batch_size x 1 x embedding_size
        tail: batch_size x 1 x embedding_size (tail-batch: batch_size x sample_size x embedding_size)

        attention = V * softmax(K^T * Q / sqrt(d))
        Q = head, K,V = relation, attention = tail
        """
        if mode == 'head-batch':
            batch_size, sample_size, embedding_size = head.size()
            Q = head.view(batch_size, sample_size, 8, 25)
            K = relation.view(batch_size, 1, 8, 25)
            V = relation2.view(batch_size, 1, 8, 25)
            K2 = relation3.view(batch_size, 1, 8, 25)
            V2 = relation4.view(batch_size, 1, 8, 25)
            Attn = tail.view(batch_size, 1, 8, 25)
            AttnQ = Q.matmul(K.transpose(2, 3)).div(self.dim).softmax(3).matmul(V) + Q
            AttnQ = AttnQ.matmul(K2.transpose(2, 3)).div(self.dim).softmax(3).matmul(V2)
            score = AttnQ - Attn
            score = score.view(batch_size, sample_size, embedding_size)
            # score = head + (relation - tail)
        else:
            batch_size, sample_size, embedding_size = tail.size()
            Q = head.view(batch_size, 1, 8, 25)
            K = relation.view(batch_size, 1, 8, 25)
            V = relation2.view(batch_size, 1, 8, 25)
            K2 = relation3.view(batch_size, 1, 8, 25)
            V2 = relation4.view(batch_size, 1, 8, 25)
            Attn = tail.view(batch_size, sample_size, 8, 25)
            AttnQ = Q.matmul(K.transpose(2, 3)).div(self.dim).softmax(3).matmul(V) + Q
            AttnQ = AttnQ.matmul(K2.transpose(2, 3)).div(self.dim).softmax(3).matmul(V2)
            score = AttnQ - Attn
            score = score.view(batch_size, sample_size, embedding_size)
            # score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def Attn3(self, head, relation, relation2, relation3, relation4, tail, mode):
        """
        head: batch_size x 1 x embedding_size (head-batch: batch_size x sample_size x embedding_size)
        relation: batch_size x 1 x embedding_size
        tail: batch_size x 1 x embedding_size (tail-batch: batch_size x sample_size x embedding_size)

        attention = V * softmax(K^T * Q / sqrt(d))
        Q = head, K,V = relation, attention = tail
        """
        if mode == 'head-batch':
            batch_size, sample_size, embedding_size = head.size()
            Q = head.view(batch_size, sample_size, 8, 25)
            K = relation.view(batch_size, 1, 8, 25)
            V = relation2.view(batch_size, 1, 8, 25)
            K2 = relation3.view(batch_size, 1, 8, 25)
            V2 = relation4.view(batch_size, 1, 8, 25)
            Attn = tail.view(batch_size, 1, 8, 25)
            AttnQ = V.matmul(K.transpose(2, 3).matmul(Q).div(self.dim).softmax(3)) + Q
            AttnQ = V2.matmul(K2.transpose(2, 3).matmul(AttnQ).div(self.dim).softmax(3))
            score = AttnQ - Attn
            score = score.view(batch_size, sample_size, embedding_size)
            # score = head + (relation - tail)
        else:
            batch_size, sample_size, embedding_size = tail.size()
            Q = head.view(batch_size, 1, 8, 25)
            K = relation.view(batch_size, 1, 8, 25)
            V = relation2.view(batch_size, 1, 8, 25)
            K2 = relation3.view(batch_size, 1, 8, 25)
            V2 = relation4.view(batch_size, 1, 8, 25)
            Attn = tail.view(batch_size, sample_size, 8, 25)
            AttnQ = V.matmul(K.transpose(2, 3).matmul(Q).div(self.dim).softmax(3)) + Q
            AttnQ = V2.matmul(K2.transpose(2, 3).matmul(AttnQ).div(self.dim).softmax(3))
            score = AttnQ - Attn
            score = score.view(batch_size, sample_size, embedding_size)
            # score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def Attn2(self, head, relation, relation2, tail, mode):
        """
        head: batch_size x 1 x embedding_size (head-batch: batch_size x sample_size x embedding_size)
        relation: batch_size x 1 x embedding_size
        tail: batch_size x 1 x embedding_size (tail-batch: batch_size x sample_size x embedding_size)

        attention = V * softmax(K^T * Q / sqrt(d))
        Q = head, K,V = relation, attention = tail
        """
        if mode == 'head-batch':
            batch_size, sample_size, embedding_size = head.size()
            Q = head.view(batch_size, sample_size, 8, 25)
            K = relation.view(batch_size, 1, 8, 25)
            V = relation2.view(batch_size, 1, 8, 25)
            Attn = tail.view(batch_size, 1, 8, 25)
            AttnQ = V.matmul(K.transpose(2, 3).matmul(Q).div(self.dim).softmax(3))
            score = AttnQ - Attn
            score = score.view(batch_size, sample_size, embedding_size)
            # score = head + (relation - tail)
        else:
            batch_size, sample_size, embedding_size = tail.size()
            Q = head.view(batch_size, 1, 8, 25)
            K = relation.view(batch_size, 1, 8, 25)
            V = relation2.view(batch_size, 1, 8, 25)
            Attn = tail.view(batch_size, sample_size, 8, 25)
            AttnQ = V.matmul(K.transpose(2, 3).matmul(Q).div(self.dim).softmax(3))
            score = AttnQ - Attn
            score = score.view(batch_size, sample_size, embedding_size)
            # score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def Attn(self, head, relation, relation2, tail, mode):
        """
        head: batch_size x 1 x embedding_size (head-batch: batch_size x sample_size x embedding_size)
        relation: batch_size x 1 x embedding_size
        tail: batch_size x 1 x embedding_size (tail-batch: batch_size x sample_size x embedding_size)

        attention = V * softmax(K^T * Q / sqrt(d))
        Q = head, K,V = relation, attention = tail
        """
        if mode == 'head-batch':
            batch_size, sample_size, embedding_size = head.size()
            Q = head.view(batch_size, sample_size, 8, 25)
            K = relation.view(batch_size, 1, 8, 25)
            V = relation2.view(batch_size, 1, 8, 25)
            Attn = tail.view(batch_size, 1, 8, 25)
            AttnQ = Q.matmul(K.transpose(2, 3)).div(self.dim).softmax(3).matmul(V)
            score = AttnQ - Attn
            score = score.view(batch_size, sample_size, embedding_size)
            # score = head + (relation - tail)
        else:
            batch_size, sample_size, embedding_size = tail.size()
            Q = head.view(batch_size, 1, 8, 25)
            K = relation.view(batch_size, 1, 8, 25)
            V = relation2.view(batch_size, 1, 8, 25)
            Attn = tail.view(batch_size, sample_size, 8, 25)
            AttnQ = Q.matmul(K.transpose(2, 3)).div(self.dim).softmax(3).matmul(V)
            score = AttnQ - Attn
            score = score.view(batch_size, sample_size, embedding_size)
            # score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):

        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    @staticmethod
    def loss(model,
             positive_sample, negative_sample, subsampling_weight, mode,
             single_mode="single"):
        negative_score = model((positive_sample, negative_sample), mode=mode)
        negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample, mode=single_mode)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        loss = (positive_sample_loss + negative_sample_loss) / 2
        return loss

    @staticmethod
    def train_step(model, optimizer,
                   positive_sample, negative_sample, subsampling_weight, mode,
                   align_positive_sample, align_negative_sample, align_subsampling_weight, align_mode,
                   device="cuda"):

        model.train()
        optimizer.zero_grad()

        positive_sample = positive_sample.to(device)
        negative_sample = negative_sample.to(device)
        subsampling_weight = subsampling_weight.to(device)
        if align_mode is not None:
            align_positive_sample = align_positive_sample.to(device)
            align_negative_sample = align_negative_sample.to(device)
            align_subsampling_weight = align_subsampling_weight.to(device)

        raw_loss = model.loss(model,
                              positive_sample, negative_sample, subsampling_weight,
                              mode, "single")
        if align_mode is not None:
            align_loss = model.loss(model,
                                    align_positive_sample, align_negative_sample, align_subsampling_weight,
                                    align_mode, "align-single")
        else:
            align_loss = raw_loss

        loss = (raw_loss + align_loss) / 3
        loss.backward()
        optimizer.step()

        return loss.item(), raw_loss.item(), align_loss.item()


# endregion


class Trainer:
    def __init__(self,
                 data: DBP15kData,
                 output: OutputSchema,

                 device="cuda",
                 learning_rate=0.001,
                 gcn=False
                 ):
        self.log = output.logger.info
        self.data = data
        self.output = output
        self.out = output.output_path
        self.learning_rate = learning_rate
        self.device = device

        for text in data.dump():
            self.log(text)

        train_dataloader_head = DataLoader(
            TrainDataset(self.data.train_triples_ids,
                         self.data.entity_count, self.data.relation_count, self.data.entity_count, 512,
                         'head-batch'),
            batch_size=512,
            shuffle=False,
            num_workers=8,
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_tail = DataLoader(
            TrainDataset(self.data.train_triples_ids,
                         self.data.entity_count, self.data.relation_count, self.data.entity_count, 512,
                         'tail-batch'),
            batch_size=512,
            shuffle=False,
            num_workers=8,
            collate_fn=TrainDataset.collate_fn
        )
        self.train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        if gcn:
            align_dataloader_head = DataLoader(
                AlignDataset(self.data.train_seeds_ids, self.data.kg1_entities_ids, self.data.kg2_entities_ids,
                             self.data.entity_count, 512,
                             "align-head-batch"),
                batch_size=512,
                shuffle=True,
                num_workers=8,
                collate_fn=AlignDataset.collate_fn
            )
            align_dataloader_tail = DataLoader(
                AlignDataset(self.data.train_seeds_ids, self.data.kg1_entities_ids, self.data.kg2_entities_ids,
                             self.data.entity_count, 512,
                             "align-tail-batch"),
                batch_size=512,
                shuffle=True,
                num_workers=8,
                collate_fn=AlignDataset.collate_fn
            )
            self.align_iterator = BidirectionalOneShotIterator(align_dataloader_head, align_dataloader_tail)

        self.model = Kert(
            self.data.train_seeds_ids,
            nentity=self.data.entity_count,
            nrelation=self.data.relation_count,
            nvalue=self.data.entity_count,
            hidden_dim=200,
            gamma=24.0,
        ).to(self.device)

        self.optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate
        )

    def run_train(self, need_to_load_checkpoint=True, gcn=False):
        self.log("start training")
        init_step = 1
        total_steps = 500001
        test_steps = 5000
        score = 0
        last_score = score

        if need_to_load_checkpoint:
            _, init_step, _ = load_checkpoint(self.model, self.optim, self.out.checkpoint_path())
            self.log("恢复模型后，查看一下模型状态")
            last_score = self.run_test()

        progbar = Progbar(max_step=total_steps)
        start_time = time.time()
        for step in range(init_step, total_steps):
            positive_sample, negative_sample, subsampling_weight, mode = next(self.train_iterator)
            align_positive_sample, align_negative_sample, align_subsampling_weight, align_mode = None, None, None, None
            if gcn and random.randint(1, 10) < 4:
                align_positive_sample, align_negative_sample, align_subsampling_weight, align_mode = next(
                    self.align_iterator)

            loss, Kert_loss, align_loss = self.model.train_step(self.model, self.optim,
                                                                positive_sample, negative_sample,
                                                                subsampling_weight, mode,
                                                                align_positive_sample, align_negative_sample,
                                                                align_subsampling_weight, align_mode)

            progbar.update(step + 1, [
                ("loss", loss),
                ("align", align_loss),
                ("Kert", Kert_loss),
                ("cost", round((time.time() - start_time)))
            ])
            if step > init_step and step % test_steps == 0:
                score = self.run_test()
                if score > last_score:
                    self.log("保存 (+%.2f%%) (%.2f%% > %.2f%%)" %
                             ((score - last_score) * 100, score * 100, last_score * 100))
                    last_score = score
                    path = self.out.checkpoint_path()
                    save_checkpoint(self.model, self.optim, 1, step, score, path)
                    path = self.out.entity_embedding_path()
                    save_entity_embedding_list(self.model.entity_embedding, path)
                path = self.out.entity_embedding_path(score)
                save_entity_embedding_list(self.model.entity_embedding, path)

    def run_test(self):
        computing_time = time.time()
        left_vec = get_vec2(self.model.entity_embedding, self.data.left_ids)
        right_vec = get_vec2(self.model.entity_embedding, self.data.right_ids)
        evaluate_result = evaluate_entity_alignment(len(self.data.test_seeds), left_vec, right_vec, metric='euclidean')
        self.log("")
        self.log("计算距离完成，用时 " + str(int(time.time() - computing_time)) + " 秒")
        self.log("属性消融实验")
        pretty_print(evaluate_result, self.log)
        score = get_score(evaluate_result)
        self.log("score = %.2f%%" % (score * 100))
        return score


@click.command()
@click.option('--recover', default=False, help='recover from last training')
@click.option('--lang', default='fr_en', help='dataset form DBP15k, choice: fr_en, ja_en, zh_en')
@click.option('--output', default='kert1',
              help='experiment name for output dir where saves weight files, embedding files and log')
@click.option('--data_enhance', default=True, help='enhance data')
@click.option('--gcn', default=False, help='align module inspired by GCN-Align')
def main(recover, lang, output, data_enhance, gcn):
    dataset = DBP15k(lang)
    cache_path = DBP15kCachePath(dataset.cache_path)
    data = DBP15kData(dataset, cache_path)
    data.preprocess_data()
    data.load_cache([
        "meta",
        "train_triples_ids",
        "left_ids",
        "right_ids",
        "test_seeds",
        "train_seeds_ids",
        "kg1_entities_ids",
        "kg2_entities_ids",
    ])
    if data_enhance:
        train_triples_ids_enhance_path = dataset.cache_path / "train_triples_ids_enhance.pkl"
        if train_triples_ids_enhance_path.exists():
            data.train_triples_ids = read_cache(train_triples_ids_enhance_path)
        else:
            data.train_triples_ids = append_align_triple(data.train_triples_ids, data.train_seeds_ids)
            cache_data(data.train_triples_ids, train_triples_ids_enhance_path)
        data.train_triples_count = len(data.train_triples_ids)
        data.triple_count = data.train_triples_count + data.test_triples_count + data.valid_triples_count

    outputSchema = OutputSchema(output)

    m = Trainer(data, outputSchema)
    m.run_train(need_to_load_checkpoint=recover, gcn=gcn)


# CUDA_VISIBLE_DEVICES=1 python main.py --data_enhance true --gcn true --lang fr_en --output ./result/Kert
if __name__ == '__main__':
    main()
