#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset


class KGEModel(nn.Module):
    def __init__(self, train_seeds, nentity, nrelation, nvalue, hidden_dim, gamma, double_entity_embedding=False,
                 double_relation_embedding=False):
        super(KGEModel, self).__init__()
        # self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.nvalue = nvalue
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )
        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim
        self.value_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim

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

        self.value_embedding = nn.Parameter(torch.zeros(nvalue, self.value_dim))
        # nn.init.normal_(self.value_embedding)
        nn.init.uniform_(
            tensor=self.value_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    def forward(self, sample, mode='single'):
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

            tail = torch.index_select(
                self.value_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
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

            tail = torch.index_select(
                self.value_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':

            head_part, tail_part = sample
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

            tail = torch.index_select(
                self.value_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        score = self.TransE(head, relation, tail, mode)

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
    def train_step(model, optimizer, train_iterator, device="cuda"):

        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        positive_sample = positive_sample.to(device)
        negative_sample = negative_sample.to(device)
        subsampling_weight = subsampling_weight.to(device)

        negative_score = model((positive_sample, negative_sample), mode=mode)
        negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        loss.backward()
        optimizer.step()

        return loss.item()

    @staticmethod
    def test_step(model, test_triples, all_true_triples, nentity, nrelation, device="cuda"):

        model.eval()

        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                nentity,
                nrelation,
                'head-batch'
            ),
            batch_size=4,
            num_workers=max(1, 1 // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                nentity,
                nrelation,
                'tail-batch'
            ),
            batch_size=4,
            num_workers=max(1, 1 // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        metrics = []
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                logs = []
                tempstep = 0

                for positive_sample, negative_sample, filter_bias, mode in test_dataset:

                    positive_sample = positive_sample.to(device)
                    negative_sample = negative_sample.to(device)
                    filter_bias = filter_bias.to(device)

                    batch_size = positive_sample.size(0)

                    score = model((positive_sample, negative_sample), mode)
                    score += filter_bias

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            'MRR': 1.0 / ranking,
                        })

                    # if step % 1000 == 0:
                    #    print('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1
                    tempstep += 1

                tempmetrics = {}
                for metric in logs[0].keys():
                    tempmetrics[metric] = sum([log[metric] for log in logs]) / len(logs)
                metrics.append(tempmetrics)

        return metrics
