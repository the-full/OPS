import os
import logging
import glob
import os.path as osp

import torch
import numpy as np
import h5py
from torch.utils.data import Dataset


from ATK.utils.transform import Compose
from ATK.utils.ops import farthest_point_sample
from ATK.utils.common import print_log, is_cuda_available


_asset_dir = os.getenv('ATK_ASSET_PATH', '')
_cache_dir = osp.join(_asset_dir, 'cache')
_data_root = osp.join(_asset_dir, 'dataset', 'modelnet40_ply_hdf5_2048')

class ModelNetHdf5(Dataset):
    def __init__(
        self,
        split='train',
        data_root=_data_root,
        transform=[],
        num_points=1024,
        uniform_sampling=True,
        cache_data=True,
    ):
        super().__init__()
        self.split            = split
        self.data_root        = data_root
        self.uniform_sampling = uniform_sampling
        self.cache_data       = cache_data
        self.num_points       = num_points

        self.transform = Compose(transform)

        self.raw_data = self.get_raw_data()
        self.data     = self.get_data()

    @property
    def record_name(self):
        record_name = f"modelnet_hdf5_{self.split}"
        if self.num_points is not None:
            record_name += f"_{self.num_points}points"
            if self.uniform_sampling:
                record_name += "_uniform"
        return record_name

    @property
    def record_path(self):
        return osp.join(_cache_dir, self.record_name)

    def get_raw_data(self):
        all_data  = []
        all_label = []
        for h5_name in glob.glob(osp.join(self.data_root, 'ply_data_%s*.h5'%self.split)):
            f = h5py.File(h5_name)
            data  = f['data'][:].astype('float32') # pyright: ignore
            label = f['label'][:].astype('int64') # pyright: ignore
            label = [label_list[0] for label_list in label]
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0).tolist()
        all_label = np.concatenate(all_label, axis=0).tolist()
        return all_data, all_label
    
    def get_data(self):
        if osp.isfile(self.record_path):
            print_log(f"Loading record: {self.record_name} ...", level=logging.DEBUG)
            data = torch.load(self.record_path)
        else:
            print_log(f"Preparing record: {self.record_name} ...", level=logging.DEBUG)
            data = []
            for xyz, label in zip(*self.raw_data):
                xyz = self.sampling(xyz).cpu().numpy()
                data.append(dict(xyz=xyz, category=label))
            if self.cache_data:
                torch.save(data, self.record_path)
        return data

    def sampling(self, xyz):
        if self.num_points is not None:
            if self.uniform_sampling:
                xyz = torch.tensor(xyz).float()
                if is_cuda_available():
                    xyz = xyz.cuda()
                xyz = xyz.unsqueeze(0)
                xyz = farthest_point_sample(xyz, self.num_points)[0]
                xyz = xyz.squeeze(0)
            else:
                xyz = xyz[: self.num_points]
        return xyz

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def __len__(self):
        return len(self.data)

    def prepare_data(self, idx):
        data_dict = self.data[idx]
        data_dict = self.transform(data_dict)
        return data_dict

