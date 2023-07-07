import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import logging,os
from ocr_by_transformer import run_epoch_dist,load_lbl2id_map,statistics_max_len_label,Recognition_Dataset,make_ocr_model

#python -m torch.distributed.launch --nproc_per_node=8 dist_train.py

# 设置分布式训练参数
world_size = 8

dist.init_process_group(backend='nccl', init_method='env://')
rank = dist.get_rank()
torch.cuda.set_device(rank)

base_data_dir = '../datas/ori/'
nrof_epochs = 5000
batch_size = 64
model_save_path = './ex1_ocr_model.pth'
# 读取label-id映射关系记录文件
lbl2id_map_path = os.path.join(base_data_dir, 'lbl2id_map.txt')
lbl2id_map, id2lbl_map = load_lbl2id_map(lbl2id_map_path)
# 统计数据集中出现的所有的label中包含字符最多的有多少字符，数据集构造gt信息需要用到
train_lbl_path = os.path.join(base_data_dir, 'gt.txt')
train_max_label_len = statistics_max_len_label(train_lbl_path)
sequence_len = train_max_label_len  # 数据集中字符数最多的一个case作为制作的gt的sequence_len
# 构造 dataloader
max_ratio = 8  # 图片预处理时 宽/高的最大值，不超过就保比例resize，超过会强行压缩
dataset = Recognition_Dataset(base_data_dir, lbl2id_map, sequence_len, max_ratio, 'train', pad=0)
sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, batch_size=80, num_workers=6, sampler=sampler)

# 创建模型
tgt_vocab = len(lbl2id_map.keys())
d_model = 512
mymodel = make_ocr_model(tgt_vocab, N=5, d_model=d_model, d_ff=2048, h=8, dropout=0.1)

#mymodel.load_state_dict(torch.load('mymodel.pth'))
mymodel.cuda(rank)
mymodel = torch.nn.parallel.DistributedDataParallel(mymodel, device_ids=[rank],find_unused_parameters=True)

from train_utils import LabelSmoothing,SimpleLossCompute
criterion = LabelSmoothing(size=tgt_vocab, padding_idx=0, smoothing=0.0)
# optimizer = optim.Adam(mymodel.parameters(), lr=0.001)  # Adam优化器，替换成您自己的优化器和学习率
optimizer = torch.optim.SGD(mymodel.parameters(), lr=0.01, momentum=0.9)
num_epochs = 1000
# 训练循环
for epoch in range(num_epochs):
    mymodel.train()
    loader.sampler.set_epoch(epoch)
    loss_compute = SimpleLossCompute(criterion, optimizer)
    train_mean_loss = run_epoch_dist(loader, mymodel, loss_compute, [rank])
    if rank == 0:
      print('----------------------%s-------------------------'%epoch)
    if rank == 0 and epoch%50==0:
        torch.save(mymodel.module.state_dict(), 'mymodel.pth')
# 保存模型

