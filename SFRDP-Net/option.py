import argparse
import json
import os
import multiprocessing  
# 从JSON文件中加载参数
def load_json_params(json_file):
    with open(json_file, 'r') as f:
        params = json.load(f)
    return params

# 假设JSON文件路径
json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
json_params = load_json_params(json_file_path)


# 初始化参数解析器

freeze_phys_epoch = 20



parser = argparse.ArgumentParser()

parser.add_argument('--train_db_path', type=str, default=json_params.get('train_db_path', ''))
parser.add_argument('--test_db_path', type=str, default=json_params.get('test_db_path', ''))
parser.add_argument('--batch_size', type=int, default=json_params.get('batch_size', 4))
parser.add_argument('--num_workers', type=int, default=json_params.get('num_workers', 4))
parser.add_argument('--crop_size', type=int, default=json_params.get('crop_size', 256))
parser.add_argument('--dpath', type=str, default=json_params.get('dpath', ''))
parser.add_argument('--train_haze_root', type=str, default=json_params.get('train_haze_root', ''))
parser.add_argument('--train_clear_root', type=str, default=json_params.get('train_clear_root', ''))
parser.add_argument('--test_haze_root', type=str, default=json_params.get('test_haze_root', ''))
parser.add_argument('--test_clear_root', type=str, default=json_params.get('test_clear_root', ''))
parser.add_argument('--train_lmdb_name', type=str, default=json_params.get('train_lmdb_name', ''))
parser.add_argument('--test_lmdb_name', type=str, default=json_params.get('test_lmdb_name', ''))
parser.add_argument('--seed', type=int, default=json_params.get('seed', 3402))
parser.add_argument('--lr_list', type=list, default=json_params.get('lr_list', [250, 375]))
parser.add_argument('--save_fre_step', type=int, default=json_params.get('save_fre_step', 5))
parser.add_argument('--test_fre_step', type=int, default=json_params.get('test_fre_step', 1))
parser.add_argument('--model_loadPath', type=str, default=json_params.get('model_loadPath', ''))
parser.add_argument('--opt_loadPath', type=str, default=json_params.get('opt_loadPath', ''))
parser.add_argument('--model_Savepath', type=str, default=json_params.get('model_Savepath', ''))
parser.add_argument('--optim_Savepath', type=str, default=json_params.get('optim_Savepath', ''))
parser.add_argument('--logdir', type=str, default=json_params.get('logdir', ''))
parser.add_argument('--total_epoch', type=int, default=json_params.get('total_epoch', 500))
parser.add_argument('--lr', type=float, default=json_params.get('lr', 0.0002))


parser.add_argument(
    '--freeze_phys_epoch',
    type=int,
    default=120,   # ⭐ 推荐 100~200 之间，延长物理模块训练时间
    help='epochs to freeze physics branch'
)


opt = parser.parse_args()

# 确保目录存在
if not os.path.exists(opt.logdir):
    os.makedirs(opt.logdir)
if not os.path.exists(opt.model_Savepath):
    os.makedirs(opt.model_Savepath)
if not os.path.exists(opt.optim_Savepath):
    os.makedirs(opt.optim_Savepath)
#if not os.path.exists(opt.dpath):
    #os.makedirs(opt.dpath)
if opt.dpath and not os.path.exists(opt.dpath):
    os.makedirs(opt.dpath)  
# 打印当前的参数配置
# 关键修改：只在主进程打印opt配置
if multiprocessing.current_process().name == "MainProcess":
    print("当前训练配置 opt:", vars(opt))
