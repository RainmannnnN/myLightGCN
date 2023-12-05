from time import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

import world


class BasicDataset(Dataset):
    """
    一个继承于基类的基础数据集
    每个数据集的index从0开始
    """
    def __init__(self):
        print("init dataset!")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItem(self, users):
        raise NotImplementedError

    def getUserNegItem(self, users):
        """
        对于大数据集来说是没有必要的，在大数据集一般不返回所有负item
        :param users:
        :return:
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        使用 torch.sparse.IntTensor来创建一个图
        A =
            |I,   R|
            |R^T, I|
        :return: 一个副对角线是relation矩阵的矩阵
        """
        raise NotImplementedError


class LastFM(BasicDataset):
    # TODO lastfm dataset to be done
    raise NotImplementedError


class Loader(BasicDataset):
    """
    default gowalla dataset
    can choose yelp2018 and amazon-book
    """
    def __init__(self, config=world.config, path="../data/gowalla"):
        # train or test
        world.cprint(f"loading[{path}]")
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, 'test': 1} # 这个因该是区分训练模式还是测试模式
        self.mode = self.mode_dict['train'] # 目前是训练模式1
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(" ") # 把每一行变成一个list
                    items = [int(i) for i in l[1:]] # 因为第一个是id，所以去掉。后面是item编号？
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    # 在trainUser和trainItem列表后面分别追加uid和items
                    # extend只能追加list
                    trainUser.extend([uid] * len(items)) # 这里形参其实就是长度为len(items)值全部为uid的list
                    trainItem.extend(items)
                    # 因为item编号是顺序的，找到了最大的编号相当于知道了有多少items。uid同理
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        # 把list变成矩阵
        # 大小为(n_user, ), (traindataSize, ), (traindataSize, )
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        # TODO 这里为什么要+1啊？
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.Graph = None
        print(f"{self.trainDataSize} interaction for training")
        print(f"{self.testDataSize} interaction for testing")
        # 这里dataset默认是gowalla，可以看parser.py
        # 稀疏性的计算其实就是在一个m * n 的矩阵中有多少交互数据
        # 注意这里的trainDataSize, n_users, m_items都是需要实现父类的方法得到的
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (user, item), 二分图
        # csr => compressed sparse row matrix
        # row = np.array([0, 0, 1, 2, 2, 2])
        # col = np.array([0, 2, 2, 0, 1, 2])
        # data = np.array([1, 2, 3, 4, 5, 6])
        # csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
        # >>> array([[1, 0, 2],
        #           [0, 0, 3],
        #           [4, 5, 6]])
        # 这里就是相当于建立了一个 n*m training的稀疏矩阵,data全是1，行和列的坐标在self.trainUser, self.trainItem里一一对应
        # 要注意这里的np.ones的dtype默认是float64
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser), (self.trainUser, self.trainItem))),
                            shape=(self.n_user, self.m_item))
        # squeeze是把所有维度为1的给去掉。如果指定了axis，但是该axis的维度不是1，就会爆异常
        # (n * m) sum=> (n, 1)这一步相当于计算每个user交互了几个商品 squeeze=> (n)
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        # (n * m) sum=> (1, m)这一步相当于计算每个item被交互了几次 squeeze=> (m)
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        # 这里就是检查一下，有没有用户或者商品是一次也没有交互过的(值为0)，那么就手动给它交互一次(值置1)
        # 这里写0.或者0都可以，写0.是因为之前np.ones的dtype默认是float64
        self.users_D[self.users_D == 0.] = 1
        self.items_D[self.items_D == 0.] = 1

        # 提前计算
        # 从0到n - 1个user
        self._allPos = self.getUserPosItem(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go!")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getUserPosItem(self, users):
        posItems = []
        for user in users:
            # 在稀疏矩阵的每一行(每一个user)，找到对应不为0的item
            # nonzero返回的是不为0的下标，一个tuple(row, col)
            # 这里只取col，因为row值都相同,为user
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def __build_test(self):
        """
        返回一个字典
        :return: dict: {user : [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            # 如果之前有就直接加，没有就先创建一个
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getSparseGraph(self):
        """

        :return: 返回一个用tensor表示的稀疏图
        """
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                # npy是numpy的二进制格式，npz是其压缩文件，储存的是数组
                pre_adj_mat = sp.load_npz(self.path + "/s_pre_adj_mat.npz")
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                start = time()
                # Dictionary Of Keys based sparse matrix
                # 创建一个(n + m) * (n + m)的矩阵
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                # Convert this matrix to List of Lists format.
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil() # 这个是relation matrix
                # 这里就是把副对角线变成R矩阵
                adj_mat[:self.n_users, self.m_items:] = R
                adj_mat[self.n_users:, :self.m_items] = R.T
                # 再变回去
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0]) 这里是加上了自连接但是代码上注释掉了

                # 这个应该是计算adjacency matrix的度
                # 把每一行的总合加起来，就是每一个节点的度，前n个是user，后面是item
                rowsum = np.array(adj_mat.sum(axis=1))
                # -0.5次方，然后展平
                d_inv = np.power(rowsum, -0.5).flatten()
                # 这一步感觉是防止有些节点的度是0，然后-0.5次方后就变成无限了，这里需要手动重新置0
                d_inv[np.isinf(d_inv)] = 0.
                # 这个d_mat 就是每个节点的degree matrix。是一个对角矩阵
                d_mat = sp.diags(d_inv)

                # A_hat = D -1/2 * A * D -1/2
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                # 到这一步成功构造出矩阵
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing{end - start}s, save norm_mat...")
                # 保存模型到对应dataset的文件夹里
                sp.save_npz(self.path + "/s_pre_adj_mat.npz", norm_adj)

                # 这个默认是false
                if self.split:
                    self.Graph = self._split_A_hat(norm_adj)
                    print("done split A_hat")
                else:
                    self.Graph = self.convert_sp_mat_to_sp_tensor(norm_adj)
                    # 在稀疏矩阵中，可能会出现同一个索引对应多个标量，coalesce是对相同索引的多个值求和
                    # 这里我不知道为啥要用这个函数。。
                    self.Graph = self.Graph.coalesce().to(world.devices)
                    print("don't split the matrix...")

        return self.Graph

    def _split_A_hat(self, A_hat):
        """
        把A_hat矩阵根据config[a_fold]的设置按行分段
        :param A_hat: symmetrically normalized matrix,A_hat = D -1/2 * A * D -1/2
        :return: 返回分好段的列表
        """
        A_hat_fold = []
        # 看下每一折有多长
        # 默认是有100折
        fold_len = (self.n_users + self.m_items) // self.folds
        for i in range(self.folds):
            start = i * fold_len
            # 判断是不是最后一折
            if i == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = start + fold_len

            A_hat_fold.append(self.convert_sp_mat_to_sp_tensor(A_hat[start:end]).coalesce().to(world.devices))
            return A_hat_fold

    def convert_sp_mat_to_sp_tensor(self, sp_mat):
        """
        把传入的sp_mat变成tensor的形式
        :param sp_mat: A_hat矩阵或者是A_hat的一部分
        :return: 返回sp_mat的tensor形式
        """
        # 首先将矩阵转换成坐标的表示形式
        coo = sp_mat.tocoo().astype(np.float32)
        # 取到行列和对应的数据
        row = torch.tensor(coo.row).long()
        col = torch.tensor(coo.col).long()
        data = torch.FloatTensor(col.data)
        # 这个其实就是把row和col叠在一起
        # index的shape为(2, len(data))
        index = torch.stack([row, col])
        # 返回的是一个矩阵的tensor表达形式
        return torch.sparse.FloatTensor(indices=index, values=data, size=torch.Size(coo.shape))

    def getUserItemFeedback(self, users, items):
        # TODO
        pass





