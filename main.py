# This is a sample Python script.
from pprint import pprint

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn

if __name__ == '__main__':

    print("=========测试concat和stack的区别===========")
    user_weight = torch.rand(5)
    item_weight = torch.rand(5)
    print(user_weight, item_weight)
    all_emb = torch.cat([user_weight, item_weight])
    print(all_emb, all_emb.size())
    all_emb = torch.stack([user_weight, item_weight])
    print(all_emb, all_emb.size())

    temp_emb = [user_weight, item_weight]
    temp_emb = torch.cat(temp_emb)
    print("temp_emb = ", temp_emb)

    first_layer_output = torch.rand(7)
    second_layer_output = torch.rand(7)
    output = [first_layer_output, second_layer_output]
    print("output is ", output)
    final_output = torch.stack(output, dim=1)
    print(final_output)
    print("size of final_output is ", final_output.shape)

    # print("=========以下是coo函数的test===========")
    # matrix = np.array([[1., 0., 0., 0., 1., 1., 1.],
    #                     [1., 0., 1., 0., 0., 0., 0.],
    #                     [1., 1., 1., 0., 0., 0., 0.],
    #                     [0., 0., 0., 0., 0., 1., 1.],
    #                     [0., 1., 0., 1., 0., 0., 1.],
    #                     [0., 1., 1., 0., 1., 0., 1.],
    #                     [1., 0., 1., 0., 0., 1., 1.]])
    # matrix = sp.csr_matrix(matrix, dtype=np.float32)
    # print(matrix)
    # coo = matrix.tocoo()
    # print(f"coo shape is {coo.shape}")
    # # 分别打印一下看这三个属性是什么
    # pprint(coo.row)
    # pprint(coo.col)
    # pprint(coo.data)
    # # 转换成tensor
    # row = torch.tensor(coo.row).long()
    # col = torch.tensor(coo.col).long()
    # data = torch.FloatTensor(coo.data)
    #
    # index = torch.stack([row, col])
    # print(f"index is {index}")
    # print()
    # result = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    # pprint(result)

    # print("=========以下是sp.dok_matrix函数的test===========")
    # matrix = sp.dok_matrix((5, 5), dtype=float)
    # for i in range(5):
    #     for j in range(5):
    #         matrix[i, j] = i + j
    #
    # matrix = matrix.tolil()
    # print(matrix)
    #
    # rowsum = np.array(matrix.sum(axis=1))
    # print(rowsum)
    # d_inv = np.power(rowsum, -0.5).flatten()
    # print(d_inv)
    # d_inv[np.isinf(d_inv)] = 0.
    # print(d_inv)
    # d_mat = sp.diags(d_inv)
    # print(d_mat)

    # random_index = torch.rand(5) + 0.6
    # print(random_index)
    # random_index = random_index.int().bool()
    # print(random_index)

    # print("=========以下是nn.embedding函数的test===========")
    # a = np.random.randint(0, 10, 8)
    # matrix = torch.from_numpy(a)
    # print(matrix)
    # print(matrix.size())
    # emb = nn.Embedding(10, 5)
    # # print(emb.weight)
    # # print(emb.sparse)
    # m_emb = emb(matrix)
    # print('m_emb = ', m_emb)
    # print(m_emb.size())
    # print(type(m_emb))
    # print(m_emb.norm(2))
    print()

    # print("=========以下是csr_matrix函数的test===========")
    # row = np.array([0, 0, 1, 2, 2, 2])
    # col = np.array([0, 2, 2, 0, 1, 3])
    # data = np.array(np.ones(6))
    # sparse_m = csr_matrix((data, (row, col)), shape=(5, 7)).toarray()
    # print(sparse_m, 'shape = ', sparse_m.shape)
    # #
    # # users_D = np.array(sparse_m.sum(axis=1)).squeeze()
    # # items_D = np.array(sparse_m.sum(axis=0)).squeeze()
    # # print('users_D = ', users_D, 'shape = ', users_D.shape)
    # # print('items_D = ', items_D,  'shape = ', items_D.shape)
    # # print('=' * 30)
    # # users_D[users_D == 0.] = 1
    # # print('users_D = ', users_D)
    # # items_D[items_D == 0.] = 1
    # # print('items_D = ', items_D)
    # print(sparse_m.nonzero())
    # print(sparse_m.nonzero()[1])

