from typing import List
import logging
import os
import os.path as osp

import torch
from torch.utils.data import Dataset
import numpy as np

from ATK.utils.transform import Compose
from ATK.utils.ops import farthest_point_sample
from ATK.utils.common import print_log, is_cuda_available


class_name_40 = [
    'airplane', 'bathtub', 'bed',      'bench',      'bookshelf',  'bottle',
    'bowl',     'car',     'chair',    'cone',       'cup',        'curtain',
    'desk',     'door',    'dresser',  'flower_pot', 'glass_box',  'guitar',
    'keyboard', 'lamp',    'laptop',   'mantel',     'monitor',    'night_stand',
    'person',   'piano',   'plant',    'radio',      'range_hood', 'sink',
    'sofa',     'stairs',  'stool',    'table',      'tent',       'toilet',
    'tv_stand', 'vase',    'wardrobe', 'xbox'
]

# class_name_10 = [
#     'bathtub', 'bed',         'chair', 'desk',  'dresser', 
#     'monitor', 'night_stand', 'sofa',  'table', 'toilet'
# ]

_asset_dir = os.getenv('ATK_ASSET_PATH', '')
_cache_dir = osp.join(_asset_dir, 'cache')
_data_root = osp.join(_asset_dir, 'dataset', 'modelnet40_normal_resampled')

class ModelNetResampled(Dataset):
    def __init__(
        self,
        split="train",
        data_root=_data_root,
        class_names=class_name_40,
        transform=None,
        num_points=1024,
        uniform_sampling=True,
        cache_data=True,
    ):
        super().__init__()
        self.data_root        = data_root
        self.split            = split
        self.class_names      = class_names
        self.num_points       = num_points
        self.uniform_sampling = uniform_sampling
        self.cache_data       = cache_data

        self.transform = Compose(transform)

        self.data_list: List[str]
        self.data_list = self.get_data_list().tolist()

        if os.path.isfile(self.record_path):
            print_log(
                f"Loading record: {self.record_name} ...",
                level=logging.INFO,
            )
            self.data = torch.load(self.record_path)
        else:
            print_log(
                f"Preparing record: {self.record_name} ...",
                level = logging.INFO
            )
            self.data = {}
            for idx in range(len(self.data_list)):
                data_name = self.data_list[idx]
                print_log(
                    f"Parsing data [{idx}/{len(self.data_list)}]: {data_name}",
                    level = logging.INFO
                )
                self.data[data_name] = self.get_data(idx)
            if cache_data:
                torch.save(self.data, self.record_path)

    @property
    def record_name(self):
        record_name = f"modelnet_resampled_{self.split}"
        if self.num_points is not None:
            record_name += f"_{self.num_points}points"
            if self.uniform_sampling:
                record_name += "_uniform"
        return record_name

    @property
    def record_path(self):
        return osp.join(_cache_dir, self.record_name)

    @property
    def class_name_idx_map(self):
        return dict(zip(self.class_names, range(len(self.class_names))))

    @torch.no_grad()
    def get_data(self, idx):
        data_idx = idx % len(self.data_list)
        data_name = self.data_list[data_idx]
        if data_name in self.data.keys():
            return self.data[data_name]
        else:
            data_shape = "_".join(data_name.split("_")[0:-1])
            data_path = os.path.join(
                self.data_root, data_shape, self.data_list[data_idx] + ".txt"
            )
            data = np.loadtxt(data_path, delimiter=",").astype(np.float32)
            if self.num_points is not None:
                if self.uniform_sampling:
                    data = torch.tensor(data).float()
                    if is_cuda_available():
                        data = data.cuda()
                    data = farthest_point_sample(data.unsqueeze(0), self.num_points)[0].squeeze(0)
                    data = data.cpu().numpy()
                else:
                    data = data[: self.num_points]
            xyz, normal = data[:, 0:3], data[:, 3:6]
            category = np.array([self.class_name_idx_map[data_shape]])
            return dict(xyz=xyz, normal=normal, category=category)

    def get_data_list(self):
        assert isinstance(self.split, str)
        split_path = os.path.join(
            self.data_root, "modelnet40_{}.txt".format(self.split)
        )
        data_list = np.loadtxt(split_path, dtype="str")
        return data_list

    def get_data_name(self, idx):
        data_idx = idx % len(self.data_list)
        return self.data_list[data_idx]

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def __len__(self):
        return len(self.data_list)

    def prepare_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

