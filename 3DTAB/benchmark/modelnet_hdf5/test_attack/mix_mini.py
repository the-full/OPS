import ATK
import os 
import os.path as osp
import random
import shutil



_asset_dir = os.getenv('ATK_ASSET_PATH', '')
_mini_dir  = osp.join(_asset_dir, 'dataset', 'modelnet40_ply_hdf5_2048_mini')
_save_dir  = osp.join(_mini_dir, 'mixin')

if not osp.exists(_save_dir):
    os.makedirs(_save_dir)

train_dir = osp.join(_mini_dir, 'train')
test_dir  = osp.join(_mini_dir, 'test')

train_list = os.listdir(train_dir)
test_list  = os.listdir(test_dir)


def parse_data(file_list, data_dir):
    cls_dict = {}
    for filename in file_list:
        cls_name = filename.split('_&_')[0] 
        if cls_name not in cls_dict: 
            cls_dict[cls_name] = [osp.join(data_dir, filename)]
        else:
            cls_dict[cls_name].append(osp.join(data_dir, filename))
    return cls_dict


train_cls_dict = parse_data(train_list, train_dir)
test_cls_dict  = parse_data(test_list, test_dir)


def mixin_data(train_cls_dict, test_cls_dict, num_per_cls):
    mixed_cls_dict = {}
    for cls_name, train_files in train_cls_dict.items():
        test_files = test_cls_dict.get(cls_name, [])
        
        total_files = train_files + test_files
        random.shuffle(total_files)
        
        mixed_cls_dict[cls_name] = total_files[:num_per_cls]
    
    return mixed_cls_dict


mixed_cls_dict = mixin_data(train_cls_dict, test_cls_dict, 25)
mixed_list = []
for cls_name in mixed_cls_dict:
    mixed_list += mixed_cls_dict[cls_name]


def save_mixin_data(mixed_list, save_dir):
    for filepath in mixed_list:
        filepath: str
        filename = osp.basename(filepath)
        split = 'train' if filepath.find(train_dir) != -1 else 'test'
        savename = split + '_' + filename
        savepath = osp.join(save_dir, savename)
        shutil.copy(filepath, savepath)

save_mixin_data(mixed_list, _save_dir)
