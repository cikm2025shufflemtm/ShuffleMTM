import argparse
import torch
from exp.exp_shufflemtm import Exp_ShuffleMTM
from data_provider.data_factory import get_input_dim
import random
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

#%%
parser = argparse.ArgumentParser(description='PMS')

parser.add_argument('--seed', type=int, default=2024, help='random seed')
# basic config
parser.add_argument('--task_name', type=str, required=True, default='pretrain', help='task name, options:[pretrain, finetune]')
parser.add_argument('--is_training', type=int, default=1, help='status')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='datasets/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# Patch
parser.add_argument('--patch_len', type=int, default=8, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride between patch')
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio')

# Model args
parser.add_argument('--e_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--d_layers', type=int, default=1, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=4, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=16, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=128, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')

# optimization
parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers') #
parser.add_argument('--itr', type=int, default=1, help='experiments times') #
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs') #
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data') #
parser.add_argument('--pretrain_learning_rate', type=float, default=1e-4, help='optimizer learning rate') #
parser.add_argument('--finetune_learning_rate', type=float, default=1e-4)
parser.add_argument('--patience', type=int, default=3, help='early stopping patience') #
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate') #
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--pct_start', type=float, default=0.3)
# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu') #
parser.add_argument('--gpu', type=int, default=0, help='gpu') # 
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus') 

## save
parser.add_argument('--checkpoints', type=str, default='./outputs/checkpoints/', help='location of model fine-tuning checkpoints')
parser.add_argument('--pretrain_checkpoints', type=str, default='./outputs/pretrain_checkpoints/', help='location of model pre-training checkpoints')
parser.add_argument('--transfer_checkpoints', type=str, default='ckpt_best.pth', help='checkpoints we will use to finetune, options:[ckpt_best.pth, ckpt10.pth, ckpt20.pth...]')
parser.add_argument('--load_checkpoints', type=str, default=None, help='location of model checkpoints')
# parser.add_argument('--select_channels', type=float, default=1, help='select the rate of channels to train')

parser.add_argument('--ca_query', type=str, default='original', choices=['permute', 'original'])
parser.add_argument('--rec', type=str, default='original', choices=['mask', 'shuffle', 'original'])
parser.add_argument('--freeze_epochs', type=int, default=10)
parser.add_argument('--finetune_epochs', type=int, default=20)
parser.add_argument('--finetune', type=str, default='freeze_finetune', choices=['finetune_only', 'freeze_finetune'])
parser.add_argument('--missing_rate', type=float, default=0)

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
print('torch.cuda.device_count()', torch.cuda.device_count())
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
print('Args in experiment:')
print(args)

args.num_patch = (max(args.seq_len, args.patch_len)-args.patch_len) // args.stride + 1
args.head_type = 'pretrain' if args.task_name=='pretrain' else 'prediction'
args.c_in = get_input_dim(args.data)


fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

Exp = Exp_ShuffleMTM
if args.task_name == 'pretrain':
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_sl{}_el{}_dl{}_dm{}_df{}_nh{}_pl{}_ep{}_bs{}_lr{}'.format(
            args.data,
            args.seq_len,
            args.e_layers,
            args.d_layers,
            args.d_model,
            args.d_ff,
            args.n_heads,
            args.patch_len,
            args.train_epochs,
            args.batch_size,
            args.pretrain_learning_rate,

        )

        exp = Exp(args)  # set experiments
        print('>>>>>>>start pre_training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.pretrain(setting)
        torch.cuda.empty_cache()
        
elif args.task_name == 'finetune':
    for ii in range(args.itr):
        # setting record of experiments
        finetune_setting = '{}_sl{}_el{}_dl{}_dm{}_df{}_nh{}_pl{}_ep{}_bs{}_plr{}_flr_{}'.format(
            args.data,
            args.seq_len,
            args.e_layers,
            args.d_layers,
            args.d_model,
            args.d_ff,
            args.n_heads,
            args.patch_len,
            args.train_epochs,
            args.batch_size,
            args.pretrain_learning_rate,
            args.finetune_learning_rate,
        )
        
        pretrain_setting = '{}_sl{}_el{}_dl{}_dm{}_df{}_nh{}_pl{}_ep{}_bs{}_lr{}'.format(
            args.data,
            args.seq_len,
            args.e_layers,
            args.d_layers,
            args.d_model,
            args.d_ff,
            args.n_heads,
            args.patch_len,
            args.train_epochs,
            args.batch_size,
            args.pretrain_learning_rate,

        )


        args.load_checkpoints = os.path.join(args.pretrain_checkpoints, args.data, pretrain_setting, str(args.seed), args.transfer_checkpoints)
        exp = Exp(args)  # set experiments
        
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(finetune_setting))
        exp.train(finetune_setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(finetune_setting))
        exp.test(finetune_setting)
        torch.cuda.empty_cache()
