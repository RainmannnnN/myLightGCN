import os
import torch
from enum import Enum
from parse import parse_arg
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_arg() # 初始化参数列表

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
print("__file__ = ", __file__) # world.py的路径
print("os.path.dirname(__file__) = ", os.path.dirname(__file__)) # world.py所在文件夹的路径
print("ROOT_PATH = ", ROOT_PATH) # 项目的路径
# 有了项目的根路径，就可以知道各个文件夹的路径了
CODE_PATH = os.path.join(ROOT_PATH, 'code')
DATA_PATH = os.path.join(ROOT_PATH, 'data')
BOARD_PATH = os.path.join(CODE_PATH, 'runs')
FILE_PATH = os.path.join(CODE_PATH, 'checkpoints')

import sys

sys.path.append(os.path.join(CODE_PATH, 'source')) # 将自己的模块添加到目录里
print("sys.path = ", sys.path) # sys.path是一个list，也就是在一个list后面加上source的绝对路径

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book']  # 允许的全部数据集，如果需要加入新数据集要在这里添加
all_models = ['mf', 'lgn']
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers'] = args.layer
config['dropout'] = args.dropout
config['keep_prob'] = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False

GPU = torch.cuda.is_available()
devices = torch.device('cuda' if GPU else 'cpu')
CORES = multiprocessing.cpu_count()
seed = args.seed

dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path

print("args.topks = ", args.topks, ", type = ", type(args.topks))
# 详情可以看eval的文档，但是不知道为什么topks这里的设定要是一个str，还要多写一步
topks = eval(args.topks) # 这里利用了eval方法的特性，把str变成了list
print("topks = ", topks, ", type = ", type(topks))
tensorboard = args.tensorboard
comment = args.comment

# 这里是用来抑制pandas的FutureWarning(对于未来特性更改的警告)的
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def cprint(words: str):
    """
    格式化输出字符串
    :param words: 字符串
    :return: 黄底字符串
    """
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
print(logo)
cprint("logo")

