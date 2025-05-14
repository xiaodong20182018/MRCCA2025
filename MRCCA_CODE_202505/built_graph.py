r"""
Interface used for dataset and batch-wise training.
"""

import os
import logging
import torch
import numpy as np

from collections import defaultdict
from ordered_set import OrderedSet
from torch.utils import data
from torch_geometric.data import Data
from torch_scatter import scatter_add


class KBDataset(data.Dataset):

    def __init__(self, triplets, num_entity, params, training=False):
        self.triplets = triplets
        self.num_entity = num_entity
        self.params = params
        self.training = training

    def collate_fn(self, batch):
        triple = torch.stack([_[0] for _ in batch], dim=0)
        triple_label = torch.stack([_[1] for _ in batch], dim=0)

        return triple, triple_label

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        # triple, label, sub_samp = torch.LongTensor(triplet['triple']), np.int32(triplet['label']), np.float32(triplet['sub_samp'])
        triple, label = torch.LongTensor(
            triplet['triple']), np.int32(triplet['label'])
        triple_label = self.get_label(label)

        if self.training is True and self.params.lbl_smooth != 0.0:
            triple_label = (1.0 - self.params.lbl_smooth) * \
                triple_label + (1.0 / self.num_entity)

        return triple, triple_label

    def get_label(self, label):
        y = np.zeros([self.num_entity], dtype=np.float32)
        for e in label:
            y[e] = 1.0
        return torch.FloatTensor(y)


class DataLoader(object):

    def __init__(self, dataset, names2ids, rels2ids):
        self.data_dir =  dataset
        self.entity2id = names2ids
        self.relation2id = rels2ids.copy()
        self.graph = self._load_data()

    def _load_data(self):

        # read entities and relations
        ent_set, rel_set = OrderedSet(), OrderedSet()
        for data_type in ['train', 'valid', 'test']:
            for line in open(os.path.join(self.data_dir, data_type + '.txt'), 'r'):
                sub, rel, obj = map(str.lower, line.strip().split())
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)
        # self.entity2id = {ent: idx for idx, ent in enumerate(ent_set)}
        # self.relation2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.relation2id.update({rel + '_reverse': idx + len(self.relation2id)
                                 for idx, rel in enumerate(self.relation2id.keys())})

        self.num_entity = len(self.entity2id)
        self.num_relation = len(self.relation2id) // 2

        # read triplets
        data = defaultdict(list)
        sr2o = defaultdict(set)
        for data_type in ['train', 'valid', 'test']:
            with open(os.path.join(self.data_dir, data_type + '.txt'), 'r') as f:
                for line in f:
                    head, relation, tail = line.strip().split()
                    sub, rel, obj = self.entity2id[head], self.relation2id[relation], self.entity2id[tail]
                    # 加入三元组对应的序号
                    data[data_type].append((sub, rel, obj))
                    # 加入首实体和关系为字典，尾实体为  和逆关系
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.num_relation)].add(sub)
                    loft = 22
                    sr2o[(sub, loft)].add(sub)

            if data_type == 'train':
                sr2o_train = {k: list(v) for k, v in sr2o.items()}
        sr2o_all = {k: list(v) for k, v in sr2o.items()}
        data = dict(data)
        self.num_edge = len(data['train'])

        self.triplets = defaultdict(list)
        # 构造正样本
        for (sub, rel), objs in sr2o_train.items():
            self.triplets['train'].append(
                {'triple': (sub, rel, -1), 'label': objs, 'sub_samp': 1})

        for data_type in ['valid', 'test']:
            for sub, rel, obj in data[data_type]:
                rel_inv = rel + self.num_relation
                self.triplets['{}_{}'.format(data_type, 'tail')].append(
                    {'triple': (sub, rel, obj), 'label': sr2o_all[(sub, rel)]})
                self.triplets['{}_{}'.format(data_type, 'head')].append(
                    {'triple': (obj, rel_inv, sub), 'label': sr2o_all[(obj, rel_inv)]})
        self.triplets = dict(self.triplets)

        graph = self._build_graph(np.arange(self.num_entity, dtype=np.int64), np.array(
            data['train'], dtype=np.int64), bi_direction=True)

        # report the dataset
        logging.info('entity={}, relation={}, train_triplets={}, valid_triplets={}, test_triplets={}'.format(
            self.num_entity, self.num_relation, len(data['train']), len(data['valid']), len(data['test'])))

        return graph

    def _edge_normal(self, edge_type, edge_index, num_entity):

        edge_type, edge_index = torch.from_numpy(edge_type), torch.from_numpy(edge_index).long()
        counts = torch.ones_like(edge_type).to(torch.float)
        # 使用 scatter_add 计算每个节点作为目标节点时，边的数量（即度数）
        # deg 会得到一个大小为 num_entity 的张量，每个元素表示对应节点的度数
        deg = scatter_add(counts, edge_index[1], dim_size=num_entity)
        # 计算每条边的归一化权重，即 1/对应目标节点的度数
        edge_norm = 1 / deg[edge_index[1]]
        # 如果计算出的归一化权重为无穷大（可能由于度数为0），将这些权重置为0
        edge_norm[torch.isinf(edge_norm)] = 0

        return edge_norm

    def _build_graph(self, graph_nodes, triplets, bi_direction=True):
        """Create a graph when given triplets
        利用三元组创建一个图

        参数:
        graph_nodes: (np.ndarray) 目标图中的节点id
        triplets: (np.ndarray) 图中的节点和关系id
        返回:
        torch_geometric.data.Data

        Args:
            graph_nodes: (np.ndarray) nodes ids in the target graph
            triplets: (np.ndarray) nodes and relation ids in the graph
        Return:
            torch_geometric.data.Data
        """
        # 将三元组分解为源节点、关系和目标节点数组
        src, rel, dst = triplets.transpose()
        # Create bi-directional graph，如果bi_direction为True，则创建双向图

        if bi_direction is True:
            # 将原始的源节点和目标节点数组合并，形成反向边
            src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
            rel = np.concatenate((rel, rel + self.num_relation))
            #增加
            src, dst = np.concatenate((src, graph_nodes)), np.concatenate((dst, graph_nodes))
            graph_nodes[:] = 22
            rel = np.concatenate((rel, graph_nodes))
        # 将源节点和目标节点堆叠起来构成边的索引
        edge_index = np.stack((src, dst))
        # 生成边的ID数组，范围从0到边的数量
        edge_ids = np.arange(edge_index.shape[1])
        # 将关系和边的ID堆叠起来构成边的属性
        edge_attr = np.stack((rel, edge_ids))

        data = Data(edge_index=torch.from_numpy(edge_index),
                    edge_attr=torch.from_numpy(edge_attr))
        data.entity = torch.from_numpy(graph_nodes)
        data.num_nodes = len(graph_nodes)
        data.edge_norm = self._edge_normal(rel, edge_index, len(graph_nodes))

        return data

    def _get_dataset(self, data_type, params):

        if data_type == 'train':
            return KBDataset(self.triplets['train'], len(self.entity2id), params, training=True)
        elif data_type in ['valid_head', 'valid_tail', 'test_head', 'test_tail']:
            return KBDataset(self.triplets[data_type], len(self.entity2id), params)
        else:
            raise ValueError('Unkown data type')

    def _create_data_loader(self, dataset, batch_size, num_workers, shuffle, drop_last=False):
        iterator = data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=max(0, num_workers),
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            drop_last=drop_last
        )

        return iterator

    def get_data_loaders(self, batch_size, num_workers, params):
        # 定义数据集标记列表，分别对应训练集、验证集（头部和尾部）和测试集（头部和尾部）
        marks = ['train', 'valid_head', 'valid_tail', 'test_head', 'test_tail']
        # 定义一个与 marks 列表相同长度的布尔列表，指示每个数据集是否在迭代时丢弃最后一个不完整批次
        drops = [False] * 5
        # 初始化一个字典，用于存储不同数据集的数据加载器
        data_iters = {}
        for mark, drop in zip(marks, drops):
            # 为每个数据集创建数据加载器
            # self._create_data_loader 是一个假设的函数，用于创建给定数据集的 DataLoader
            # self._get_dataset 是一个假设的函数，用于获取指定标记的数据集
            data_iters[mark] = self._create_data_loader(self._get_dataset(mark, params),
                                                        batch_size=batch_size,      # 设置批次大小
                                                        num_workers=num_workers,    # 设置工作进程数量
                                                        shuffle=True,               # 启用随机洗牌
                                                        drop_last=drop)             # 根据 drop 决定是否丢弃最后不完整的批次
        # 返回一个包含所有数据加载器的字典
        return data_iters
