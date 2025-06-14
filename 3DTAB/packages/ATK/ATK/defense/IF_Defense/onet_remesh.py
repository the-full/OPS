import os.path as osp
import numpy as np

import torch
import trimesh
import tqdm
from omegaconf import DictConfig

from .utils.generation import Generator3D
from .utils.config_utils import load_config
from .models import get_onet

from ..DropPoint import SORDefense
from ..basic_defense import BasicDefense


_cur_dir = osp.dirname(osp.abspath(__file__))
_cfg_dir = osp.join(_cur_dir, 'configs', 'onet_config')
cfg_path = osp.join(_cfg_dir, 'onet_mn40.yaml')
dft_path = osp.join(_cfg_dir, 'default.yaml')
ckpt_path = osp.join(_cur_dir, 'pretrain', 'onet.pth')
_sor_cfg = DictConfig(dict(
    k=2,
    alpha=1.1,
))

class ONetMesh(BasicDefense):
    def __init__(
        self, 
        sample_npoint=1024,
        padding_scale=0.9,
        sor_cfg=_sor_cfg,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sample_npoint = sample_npoint
        self.padding_scale = padding_scale

        if sor_cfg is not None:
            self.sor = True
            self.sor_k = sor_cfg.k
            self.sor_alpha = sor_cfg.alpha
        else:
            self.sor = False

        cfg  = load_config(cfg_path, dft_path)
        onet = get_onet(cfg, self.device, None).eval()
        onet.load_state_dict(torch.load(ckpt_path))
        for p in onet.parameters():
            p.requires_grad = False

        self.cfg = cfg
        self.threshold = cfg['test']['threshold']
        self.input_npoint = cfg['data']['pointcloud_n']
        self.generator = self.get_generator(onet, cfg, self.device)

    def sor_process(self, pc):
        """Use SOR to pre-process pc.
        Inputs:
            pc: [N, K, 3]

        Returns list of [K_i, 3]
        """
        N = len(pc)
        batch_size = 32
        sor_pc = []
        sor_defense = SORDefense(k=self.sor_k, alpha=self.sor_alpha)

        for i in range(0, N, batch_size):
            input_pc = pc[i:i + batch_size]  # [B, K, 3]
            output_pc = sor_defense.outlier_removal(input_pc)
            # to np array list
            output_pc = [
                one_pc.detach().cpu().numpy().astype(np.float32) 
                for one_pc in output_pc
            ]
            sor_pc.append(output_pc)

        pc = []
        for i in range(len(sor_pc)):
            pc += sor_pc[i]  # list of [k, 3]

        assert len(pc[0].shape) == 2 and pc[0].shape[1] == 3
        return pc

    @staticmethod
    def preprocess_pc(pc, num_points=None, padding_scale=1.):
        # normalize into unit cube
        center = np.mean(pc, axis=0)  # [3]
        centered_pc = pc - center
        max_dim = np.max(centered_pc, axis=0)  # [3]
        min_dim = np.min(centered_pc, axis=0)  # [3]
        scale = (max_dim - min_dim).max()
        scaled_centered_pc = centered_pc / scale * padding_scale

        # select a subset as ONet input
        if num_points is not None and \
                scaled_centered_pc.shape[0] > num_points:
            idx = np.random.choice(
                scaled_centered_pc.shape[0], 
                num_points,
                replace=False
            )
            pc = scaled_centered_pc[idx]
        else:
            pc = scaled_centered_pc

        # to torch tensor
        torch_pc = torch.from_numpy(pc).float().cuda().unsqueeze(0)
        return torch_pc


    def reconstruct_mesh(self, pc, padding_scale):
        '''Reconstruct a mesh from input point cloud.
        With potentially pre-processing and post-processing.
        '''
        # pre-process
        # only use coordinates information
        pc = pc[:, :3]
        pc = self.preprocess_pc(
            pc, 
            num_points=self.input_npoint,
            padding_scale=padding_scale
        )

        # ONet mesh generation
        # shape latent code, [B, c_dim (typically 512)]
        c = self.generator.model.encode_inputs(pc)

        # z is of no use
        z = self.generator.model.get_z_from_prior(
            (1,), sample=self.generator.sample
        ).cuda()

        mesh = self.generator.generate_from_latent(z, c)
        return mesh

    def resample_points(self, ori_pc, re_mesh, num_points):
        '''Apply reconstruction and re-sampling.'''
        # sample points from it
        try:
            pc, _ = trimesh.sample.sample_surface(       # type: ignore
                re_mesh,
                count=num_points,
            )
        # reconstruction might fail
        # random sample some points as defense results
        except IndexError:
            pc = np.zeros((num_points, 3), dtype=np.float32)
            if ori_pc.shape[0] > num_points:
                # apply SRS
                idx = np.random.choice(
                    ori_pc.shape[0], 
                    num_points,
                    replace=False
                )
                pc = ori_pc[idx]
            else:
                pc[:ori_pc.shape[0]] = ori_pc
        return pc

    @staticmethod
    def normalize_pc(points):
        """points: [K, 3]"""
        points = points - np.mean(points, axis=0)[None, :]  # center
        dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
        points = points / dist  # scale
        return points

    def defense(self, adv_pcs, labels, **kwargs):
        # possible SOR preprocess
        if self.sor:
            with torch.no_grad():
                adv_pcs = self.sor_process(adv_pcs) # NOTE: return list
        torch.cuda.empty_cache()

        # reconstruct, re-sample
        re_pcs = np.zeros(
            (len(adv_pcs), self.sample_npoint, 3),
            dtype=np.float32,
        )
        #with tqdm.trange(len(adv_pcs), desc='ONet-Remesh') as pbar:
        with tqdm.trange(
            len(adv_pcs), 
            desc='ONet-Optimize',
            leave=False,
        ) as pbar:
            for i in pbar:
                one_pc  = adv_pcs[i]
                re_mesh = self.reconstruct_mesh(one_pc, self.padding_scale)
                import ipdb; ipdb.set_trace()
                re_pc   = self.resample_points(one_pc, re_mesh, self.sample_npoint)
                re_pcs[i] = self.normalize_pc(re_pc)

        return torch.tensor(re_pcs).to(self.device)

    def get_generator(self, onet, cfg, device, **kwargs):
        ''' Returns the generator object.

        Args:
            model (nn.Module): Occupancy Network model
            cfg (dict): imported yaml config
            device (device): pytorch device
        '''
        generator = Generator3D(
            onet,
            device=device,
            threshold=cfg['test']['threshold'],
            resolution0=cfg['generation']['resolution_0'],
            upsampling_steps=cfg['generation']['upsampling_steps'],
            sample=cfg['generation']['use_sampling'],
            refinement_step=cfg['generation']['refinement_step'],
            simplify_nfaces=cfg['generation']['simplify_nfaces'],
            preprocessor=None,
        )
        return generator
