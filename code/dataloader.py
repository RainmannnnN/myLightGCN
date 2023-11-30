import numpy as np
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
        :return:
        """
        raise NotImplementedError

class LastFM(BasicDataset):
    pass

class Loader(BasicDataset):
    """
    gowalla dataset
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
        # TODO 写到这里


    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize








