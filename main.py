# This is a sample Python script.
import numpy as np
from scipy.sparse import csr_matrix

if __name__ == '__main__':
    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 3])
    data = np.array(np.ones(6))
    sparse_m = csr_matrix((data, (row, col)), shape=(5, 7)).toarray()
    print(sparse_m, 'shape = ', sparse_m.shape)
    #
    # users_D = np.array(sparse_m.sum(axis=1)).squeeze()
    # items_D = np.array(sparse_m.sum(axis=0)).squeeze()
    # print('users_D = ', users_D, 'shape = ', users_D.shape)
    # print('items_D = ', items_D,  'shape = ', items_D.shape)
    # print('=' * 30)
    # users_D[users_D == 0.] = 1
    # print('users_D = ', users_D)
    # items_D[items_D == 0.] = 1
    # print('items_D = ', items_D)
    print(sparse_m.nonzero())
    print(sparse_m.nonzero()[1])

