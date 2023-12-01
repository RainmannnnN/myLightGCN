import numpy as np
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
        :return:
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







