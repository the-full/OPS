import os 
import os.path as osp

import torch
from torch.utils.data import DataLoader

from ATK import register_table


_asset_dir = os.getenv('ATK_ASSET_PATH', '')
_save_dir  = osp.join(_asset_dir, 'dataset', 'modelnet40_ply_hdf5_2048_mini')

class_name_40 = [
    'airplane', 'bathtub', 'bed',      'bench',      'bookshelf',  'bottle',
    'bowl',     'car',     'chair',    'cone',       'cup',        'curtain',
    'desk',     'door',    'dresser',  'flower_pot', 'glass_box',  'guitar',
    'keyboard', 'lamp',    'laptop',   'mantel',     'monitor',    'night_stand',
    'person',   'piano',   'plant',    'radio',      'range_hood', 'sink',
    'sofa',     'stairs',  'stool',    'table',      'tent',       'toilet',
    'tv_stand', 'vase',    'wardrobe', 'xbox'
]

def generate_adv_sample_part(
    split='train', 
    num_workers=4,
):
    dataset = register_table.datasets['ModelNetHdf5'](
        split = split,
        transform = []
    )
    data_loader = DataLoader(
        dataset, 
        num_workers=num_workers,
        batch_size=50,
        shuffle=False,
        drop_last=False
    )

    save_dir = osp.join(_save_dir, split)
    os.makedirs(save_dir, exist_ok=True)

    label_counts = {}
    for data_dict in data_loader:
        batch_size = data_dict['xyz'].shape[0]
        for i in range(batch_size):
            label_id = data_dict['category'][i]
            label_name = class_name_40[label_id]

            if label_name not in label_counts:
                label_counts[label_name] = 1
            else:
                if label_counts[label_name] == 30:
                    continue
                label_counts[label_name] += 1
            
            filename = f"{label_name}_&_{label_counts[label_name]}.pth"
            save_path = osp.join(save_dir, filename)
            save_data = dict(
                xyz = data_dict['xyz'][i],
                category = data_dict['category'][i]
            )
            torch.save(save_data, save_path)

if __name__ == "__main__":
    generate_adv_sample_part()
