import os

import torch

from ocr_by_transformer import judge_is_correct,greedy_decode,load_lbl2id_map,statistics_max_len_label,Recognition_Dataset,make_ocr_model
from torch.utils.data import DataLoader

base_data_dir = '../datas/ori/'
nrof_epochs = 1000
batch_size = 64

lbl2id_map_path = os.path.join(base_data_dir, 'lbl2id_map.txt')
lbl2id_map, id2lbl_map = load_lbl2id_map(lbl2id_map_path)
# 统计数据集中出现的所有的label中包含字符最多的有多少字符，数据集构造gt信息需要用到
train_lbl_path = os.path.join(base_data_dir, 'gt.txt')
train_max_label_len = statistics_max_len_label(train_lbl_path)
sequence_len = train_max_label_len  # 数据集中字符数最多的一个case作为制作的gt的sequence_len
# 构造 dataloader
max_ratio = 8  # 图片预处理时 宽/高的最大值，不超过就保比例resize，超过会强行压缩
dataset = Recognition_Dataset(base_data_dir, lbl2id_map, sequence_len, max_ratio, 'train', pad=0)
loader = DataLoader(dataset, batch_size=80, num_workers=0)

tgt_vocab = len(lbl2id_map.keys())
d_model = 512
mymodel = make_ocr_model(tgt_vocab, N=5, d_model=d_model, d_ff=2048, h=8, dropout=0.1)
mymodel.cuda()
import multiprocessing
multiprocessing.freeze_support()
checkpoint = torch.load('mymodel.pth', map_location=lambda storage, loc: storage)  # 加载模型参数文件
mymodel.load_state_dict(checkpoint)


mymodel.eval()
print("\n------------------------------------------------")
print("greedy decode trainset")
total_img_num = 0
total_correct_num = 0
for batch_idx, batch in enumerate(loader):
    img_input, encode_mask, decode_in, decode_out, decode_mask, ntokens = batch
    img_input = img_input.cuda()
    encode_mask = encode_mask.cuda()

    bs = img_input.shape[0]
    for i in range(bs):
        cur_img_input = img_input[i].unsqueeze(0)
        cur_encode_mask = encode_mask[i].unsqueeze(0)
        cur_decode_out = decode_out[i]

        pred_result = greedy_decode(mymodel, cur_img_input, cur_encode_mask, max_len=sequence_len, start_symbol=1,
                                    end_symbol=2)
        pred_result = pred_result.cpu()

        is_correct = judge_is_correct(pred_result, cur_decode_out)
        total_correct_num += is_correct
        total_img_num += 1
        # if not is_correct:
        #     # 预测错误的case进行打印
        #     print("----")
        #     print(cur_decode_out)
        #     print(pred_result)
total_correct_rate = total_correct_num / total_img_num * 100
print(f"total correct rate of trainset: {total_correct_rate}%")