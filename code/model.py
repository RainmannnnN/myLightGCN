from torch import nn
import torch

from world import cprint
from dataloader import BasicDataset


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PureMF(BasicModel):
    """
    矩阵分解模型
    """
    def __init__(self, config: dict, dataset: BasicDataset):
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)
        print("初始化PureMF，使用标准分布，均值为0方差为1")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def forward(self, users, items):
        # 想知道这里users和items的形状是什么啊？
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        # 每个元素相乘然后相加得到分数的总合
        scores = torch.sum(users_emb * items_emb, dim=1)
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        # softplus(x) = log(1 + exp(x))是一个激活函数
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        # 这是L2正则化
        reg_loss = ((1 / 2) * (users_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) /
                    float(len(users)))
        return loss, reg_loss


class LightGCN(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            cprint('use NORMAL distribution init')
        else:
            # config里面没有user_emb和item_emb啊？这是从哪里加载来的？
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            cprint('user PRETRAINED data')
        self.f = nn.Sigmoid()
        # 得到user_item的关系矩阵
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

    def bpr_loss(self):
        raise NotImplementedError

    def __dropout_x(self, x, keep_prob):
        """
        实现dropout的主要函数
        :param x: 是整个A_hat矩阵或者是其一部分
        :param keep_prob: the batch size for bpr loss training procedure(default=0.6)
        :return:
        """
        size = x.size()
        index = x.indices().t()
        values = x.values()
        # 这里是决定哪一些index舍去，哪一些留下来
        random_index = torch.rand(len(values)) + keep_prob
        # 大于1的为True，其余为False
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        """
        这是实现dropout的大函数，具体的实现还需要在dropout_x上实现
        :param keep_prob: the batch size for bpr loss training procedure(default=0.6)
        :return:
        """
        if self.A_split:
            # A_split == True 说明将矩阵分成了多份
            # 把每一份存在list里面，Graph就是一个list
            graph = []
            for g in self.Graph:
                # 取出每一份
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            # 若没有分割就直接把整个矩阵放进去
            graph = self.__dropout_x(self.Graph, keep_prob)

        return graph

    def computer(self):
        """
        propagate method for LightGCN
        :return: scores list of users and items
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        # 拼接在一起
        all_emb = torch.cat([users_emb, items_emb])
        # (n + m, 1)
        embs = [all_emb]

        # 只有在dropout为true且在训练时才需要使用dropout
        if self.config['dropout']:
            if self.training:
                print("dropping")
                # 这里得到的g_dropped是一个列表，里面是分段的A_hat矩阵
                g_dropped = self.__dropout(self.keep_prob)
            else:
                g_dropped = self.Graph
        else:
            g_dropped = self.Graph

        # 对每一层的GCN
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_dropped)):
                    # (fold_len, m + n) * (n + m, 1) => (fold_len, 1)
                    # 也就是说到最后temp_emb里面有self.folds个(fold_len, 1)的list
                    temp_emb.append(torch.sparse.mm(g_dropped[f], all_emb))
                # 把list里面的结果最后都concat起来
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # 这是稀疏矩阵的乘法,args1为稀疏矩阵，用法和torch.mm类似
                # (n + m, n + m) * (n + m, 1) => (n + m, 1)
                all_emb = torch.sparse.mm(g_dropped, all_emb)

            # embs是一个list，list里面是每一层最后的embedding
            # list的长度为1 + self.n_layers
            # 每一个元素是一个embedding，长度为n + m
            embs.append(all_emb)

        # 这里embs变成了tensor，size为(n + m, self.n_layers + 1)
        # 每一列是一个层的output
        embs = torch.stack(embs, dim=1)
        # 最后的输出是取平均(n + m, 1)
        light_out = torch.mean(embs, dim=1)
        # 把输出的前n个给users，后m个给item
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        raise NotImplementedError

    def getEmbedding(self, users, pos_items, neg_items):
        raise NotImplementedError

    def forward(self, users, items):
        raise NotImplementedError