# -*- encoding: utf-8 -*-
'''
@Author  :   wangjian
'''

import argparse
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()

parser.add_argument('--stage_1_outputdir', default=os.path.join(CUR_DIR,'out_1'),type=str, help='模型保存位置')
parser.add_argument('--stage_2_outputdir', default=os.path.join(CUR_DIR,'out_2'),type=str, help='模型保存位置')
parser.add_argument("--bert_pretrained_model", default='/home/wangjian0110/myWork/chinese_wwm_ext_L-12_H-768_A-12',type=str, help='bert预训练模型位置')
parser.add_argument('--task_name', default='lcqmc_pair', type=str)
parser.add_argument('--do_lower_case', default=True, action='store_false')

parser.add_argument('--do_train', default=False, action='store_true', help='是否训练模型')
parser.add_argument('--do_eval', default=False, action='store_true', help='是否eval')
parser.add_argument('--do_predict', default=False, action='store_true', help='是否predict')


parser.add_argument("--max_seq_length", default=128, type=int, help='句子最大长度')
parser.add_argument("--log_steps", default=1000, type=int)
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--eval_batch_size", default=64, type=int)
parser.add_argument("--num_train_epochs", default=10, type=int)
parser.add_argument("--warmup_proportion", default=0.1, type=float)
parser.add_argument("--learning_rate", default=2e-5, type=float)
parser.add_argument("--predict_batch_size", default=1024, type=int)

parser.add_argument("--replace_rate_prob", default=0.5, type=float)
parser.add_argument("--finetune_suc", default=False, action='store_true')
parser.add_argument("--suc_layers", default=6, type=int)


parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(CUR_DIR),'my_albert/lcqmc'))

args = parser.parse_args()
