# This is a sample Python script.
import numpy as np
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn

if __name__ == '__main__':
    print("=========以下是nn.embedding函数的test===========")
    a = np.random.randint(0, 10, 8)
    matrix = torch.from_numpy(a)
    print(matrix)
    print(matrix.size())
    emb = nn.Embedding(10, 5)
    # print(emb.weight)
    # print(emb.sparse)
    m_emb = emb(matrix)
    print('m_emb = ', m_emb)
    print(m_emb.size())
    print(type(m_emb))
    print(m_emb.norm(2))



    print("=========以下是csr_matrix函数的test===========")
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

