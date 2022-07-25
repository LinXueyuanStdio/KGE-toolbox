import random
from typing import List, Tuple, Dict, Set

import torch
from torch.utils.data import Dataset

from toolbox.data.functional import build_map_hr_t, with_inverse_relations


class ScoringAllDataset(Dataset):
    def __init__(self, train_triples_ids: List[Tuple[int, int, int]], entity_count: int):
        self.hr_t = build_map_hr_t(train_triples_ids)
        self.hr_pairs = list(self.hr_t.keys())
        self.entity_count = entity_count

    def __len__(self):
        return len(self.hr_pairs)

    def __getitem__(self, idx):
        h, r = self.hr_pairs[idx]
        data = torch.zeros(self.entity_count).float()
        data[list(self.hr_t[(h, r)])] = 1.
        h = torch.LongTensor([h])
        r = torch.LongTensor([r])
        return h, r, data


class ScoringNegativeDataset(Dataset):
    def __init__(self, train_triples_ids: List[Tuple[int, int, int]], entity_count: int, negative_sample_size: int = 32):
        self.hr_t = build_map_hr_t(train_triples_ids)
        self.hr_pairs = list(self.hr_t.keys())
        self.entity_count = entity_count
        self.negative_sample_size = negative_sample_size

    def __len__(self):
        return len(self.hr_pairs)

    def __getitem__(self, idx):
        h, r = self.hr_pairs[idx]
        data = torch.zeros(self.entity_count).float()
        data[list(self.hr_t[(h, r)])] = 1.
        idx = torch.randperm(self.entity_count)[:self.negative_sample_size]
        targets = data[idx]
        h = torch.LongTensor([h])
        r = torch.LongTensor([r])
        return h, r, idx, targets


class ScoringOneVsNegativeDataset(Dataset):
    def __init__(self, train_triples_ids: List[Tuple[int, int, int]], max_relation_id: int, entity_count: int, negative_sample_size: int = 32):
        self.train_triples_ids = train_triples_ids
        train_triples, _, _ = with_inverse_relations(train_triples_ids, max_relation_id)
        self.hr_t = build_map_hr_t(train_triples)
        self.entity_count = entity_count
        self.max_relation_id = max_relation_id
        self.negative_sample_size = negative_sample_size

    def __len__(self):
        return len(self.train_triples_ids)

    def __getitem__(self, idx):
        h, r, t = self.train_triples_ids[idx]
        reverse_r = self.max_relation_id + r
        t_sample, t_target = self.sampling(h, r, t)
        h_sample, h_target = self.sampling(t, reverse_r, h)
        h = torch.LongTensor([h])
        r = torch.LongTensor([r])
        t = torch.LongTensor([t])
        reverse_r = torch.LongTensor([reverse_r])
        return h, r, t_sample, t_target, t, reverse_r, h_sample, h_target

    def sampling(self, h, r, answer_idx):
        valid_negative = list(set(range(self.entity_count)) - self.hr_t[(h, r)])
        sample_negative = random.choices(valid_negative, k=self.negative_sample_size)
        sample_idx = torch.LongTensor([answer_idx] + sample_negative)  # (1, 1 + num_negative)
        target = torch.zeros(1 + self.negative_sample_size).float()  # (1, 1 + num_negative)
        target[0] = 1
        return sample_idx, target


class ComplementaryScoringAllDataset(Dataset):
    def __init__(self, hr_t: Dict[Tuple[int, int], Set[int]], all_keys: List[Tuple[int, int]], entity_count: int):
        self.hr_t = hr_t
        self.hr_pairs = all_keys
        self.entity_count = entity_count

    def __len__(self):
        return len(self.hr_pairs)

    def __getitem__(self, idx):
        h, r = self.hr_pairs[idx]
        data = torch.ones(self.entity_count).float()
        value = list(self.hr_t[(h, r)])
        if len(value) > 0:
            data[value] = 0.
        h = torch.LongTensor([h])
        r = torch.LongTensor([r])
        return h, r, data


class BidirectionalScoringAllDataset(Dataset):
    def __init__(self, test_triples_ids: List[Tuple[int, int, int]], hr_t: Dict[Tuple[int, int], Set[int]], max_relation_id: int, entity_count: int):
        """
        test_triples_ids: without reverse r
        hr_t: all hr->t, MUST with reverse r
        """
        self.test_triples_ids = test_triples_ids
        self.hr_t = hr_t
        self.entity_count = entity_count
        self.max_relation_id = max_relation_id

    def __len__(self):
        return len(self.test_triples_ids)

    def __getitem__(self, idx):
        h, r, t = self.test_triples_ids[idx]
        reverse_r = r + self.max_relation_id

        predict_for_hr = torch.zeros(self.entity_count).float()
        predict_for_hr[list(self.hr_t[(h, r)])] = 1.

        predict_for_tReverser = torch.zeros(self.entity_count).float()
        predict_for_tReverser[list(self.hr_t[(t, reverse_r)])] = 1.

        h = torch.LongTensor([h])
        r = torch.LongTensor([r])
        t = torch.LongTensor([t])
        reverse_r = torch.LongTensor([reverse_r])

        return h, r, predict_for_hr, t, reverse_r, predict_for_tReverser
