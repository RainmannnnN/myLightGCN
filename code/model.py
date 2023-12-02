from torch import nn
import torch
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
    pass