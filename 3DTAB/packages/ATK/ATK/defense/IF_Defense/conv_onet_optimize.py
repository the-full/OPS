import os.path as osp

import torch
import torch.nn.functional as F
import tqdm
import numpy as np
from omegaconf import DictConfig

from .utils.generation import Generator3D
from .utils.repulsion_loss import repulsion_loss
from .utils.config_utils import load_config
from .models import get_conv_onet

from ..DropPoint import SORDefense
from ..basic_defense import BasicDefense


_cur_dir = osp.dirname(osp.abspath(__file__))
_cfg_dir = osp.join(_cur_dir, 'configs', 'conv_onet_config')
cfg_path = osp.join(_cfg_dir, 'convonet_3plane_mn40.yaml')
dft_path = osp.join(_cfg_dir, 'default.yaml')
ckpt_path = osp.join(_cur_dir, 'pretrain', 'convonet.pth')
_sor_cfg = DictConfig(dict(
    k=2,
    alpha=1.1,
))

class ConvONetOpt(BasicDefense):
    def __init__(
        self, 
        sample_npoint=1024,
        init_sigma=0.01,
        padding_scale=0.9,
        iterations=200,
        batch_size=192,
        attack_lr=0.001,
        rep_weight=500.,
        sor_cfg=_sor_cfg,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sample_npoint = sample_npoint
        self.init_sigma = init_sigma
        self.padding_scale = padding_scale
        self.iterations = iterations
        self.batch_size = batch_size
        self.lr = attack_lr
        self.rep_weight = rep_weight

        if sor_cfg is not None:
            self.sor = True
            self.sor_k = sor_cfg.k
            self.sor_alpha = sor_cfg.alpha
        else:
            self.sor = False

        cfg  = load_config(cfg_path, dft_path)
        conv_onet = get_conv_onet(cfg, self.device, None).eval()
        conv_onet.load_state_dict(torch.load(ckpt_path))
        for p in conv_onet.parameters():
            p.requires_grad = False

        self.cfg = cfg
        self.threshold = cfg['test']['threshold']
        self.input_npoint = cfg['data']['pointcloud_n']
        self.generator = self.get_generator(conv_onet, cfg, self.device)

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
                one_pc.detach().cpu().numpy().
                astype(np.float32) for one_pc in output_pc
            ]
            sor_pc.append(output_pc)
        pc = []
        for i in range(len(sor_pc)):
            pc += sor_pc[i]  # list of [k, 3]
        assert len(pc[0].shape) == 2 and pc[0].shape[1] == 3
        return pc

    @staticmethod
    def preprocess_pc(pc, num_points=None, padding_scale=1.):
        """Center and scale to be within unit cube.
        Inputs:
            pc: np.array of [K, 3]
            num_points: pick a subset of points as OccNet input.
            padding_scale: padding ratio in unit cube.
        """
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
                scaled_centered_pc.shape[0], num_points,
                replace=False)
            pc = scaled_centered_pc[idx]
        else:
            pc = scaled_centered_pc

        # to torch tensor
        # torch_pc is ONet input, have fixed number of points
        # torch_all_pc is for initializing the defense point cloud
        torch_pc = torch.from_numpy(pc).\
            float().cuda().unsqueeze(0)
        torch_all_pc = torch.from_numpy(scaled_centered_pc).\
            float().cuda().unsqueeze(0)
        return torch_all_pc, torch_pc

    def init_points(self, pc):
        """Initialize points to be optimized.

        Args:
            pc (tensor): input (adv) pc, [B, N, 3]
        """
        with torch.no_grad():
            B = len(pc)

            if isinstance(pc, list):  # after SOR
                idx = [
                    torch.randint(0, len(one_pc), (self.sample_npoint,)).long().cuda() 
                    for one_pc in pc
                ]
            else:
                idx = torch.randint(0, pc.shape[1], (B, self.sample_npoint)).long().cuda()

            points = torch.stack([pc[i][idx[i]] for i in range(B)], dim=0).float().cuda()

            # add noise
            noise = torch.randn_like(points) * self.init_sigma
            points = torch.clamp(
                points + noise,
                min=-0.5 * self.padding_scale,
                max= 0.5 * self.padding_scale
            )
        return points

    @staticmethod
    def normalize_batch_pc(points):
        """points: [batch, K, 3]"""
        centroid = torch.mean(points, dim=1)  # [batch, 3]
        points -= centroid[:, None, :]  # center, [batch, K, 3]
        dist = torch.sum(points ** 2, dim=2) ** 0.5  # [batch, K]
        max_dist = torch.max(dist, dim=1)[0]  # [batch]
        points /= max_dist[:, None, None]
        return points

    def optimize_points(
        self,
        opt_points, 
        z, 
        c,
        rep_weight=1.,
        iterations=1000,
        printing=False
    ):
        """Optimization process on point coordinates.

        Args:
            opt_points (tensor): input init points to be optimized
            z (tensor): latent code
            c (tensor): feature vector
            iterations (int, optional): opt iter. Defaults to 1000.
            printing (bool, optional): print info. Defaults to False.
        """
        # 2 losses in total
        # Geo-aware loss enforces occ_value = occ_threshold by BCE
        # Dist-aware loss pushes points uniform by repulsion loss
        opt_points = opt_points.float().cuda()
        opt_points.requires_grad_()
        B, K = opt_points.shape[:2]

        # GT occ for surface
        with torch.no_grad():
            occ_threshold = torch.ones((B, K)).float().cuda() * self.threshold

        opt = torch.optim.Adam([opt_points], lr=self.lr)

        # start optimization
        for i in range(iterations + 1):
            # 1. occ = threshold
            occ_value = self.generator.model.decode(opt_points, c).logits
            occ_loss = F.binary_cross_entropy_with_logits(
                occ_value, occ_threshold, reduction='none')  # [B, K]
            occ_loss = torch.mean(occ_loss)
            occ_loss = occ_loss * K

            # 2. repulsion loss
            rep_loss = torch.tensor(0.).float().cuda()
            if rep_weight > 0.:
                rep_loss = repulsion_loss(opt_points)  # [B]
                rep_loss = torch.mean(rep_loss)
                rep_loss = rep_loss * rep_weight

            loss = occ_loss + rep_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            if printing and i % 100 == 0:
                print('iter {}, loss {:.4f}'.format(i, loss.item()))
                print('occ loss: {:.4f}, '
                    'rep loss: {:.4f}\n'
                    'occ value mean: {:.4f}'.
                    format(occ_loss.item(),
                            rep_loss.item(),
                            torch.sigmoid(occ_value).mean().item()))
        opt_points.detach_()
        opt_points = self.normalize_batch_pc(opt_points)
        return opt_points.detach().cpu().numpy()

    def defense(self, adv_pcs, labels, **kwargs):
        """Apply defense to input point clouds.

        Args:
            adv_pcs (tensor): [num_data, K, 3]
        """
        pc = adv_pcs

        # possible SOR preprocessor
        if self.sor:
            with torch.no_grad():
                pc = self.sor_process(pc)
        torch.cuda.empty_cache()

        opt_pc = np.zeros((len(adv_pcs), self.sample_npoint, 3), dtype=np.float32)

        # batch process
        with tqdm.trange(
            0, len(pc), self.batch_size, 
            desc='ConvONet-Optimize',
            leave=False,
        ) as pbar:
            for idx in pbar:
                # prepare for input
                with torch.no_grad():
                    batch_pc = pc[idx:idx + self.batch_size]  # [B, K, 3]
                    # preprocess
                    batch_proc_pc = [
                        self.preprocess_pc(
                            one_pc, 
                            num_points=self.input_npoint,
                            padding_scale=self.padding_scale
                        ) 
                        for one_pc in batch_pc
                    ]
                    # the selected input_n points from batch_pc after preprocess
                    # sel_pc are for ONet input and have fixed number of points
                    batch_proc_sel_pc = torch.cat([
                        one_pc[1] for one_pc in batch_proc_pc
                    ], dim=0).float().cuda()
                    # proc_pc may have different num_points because of SOR
                    # they're used for initializing the defense point clouds
                    try:
                        batch_proc_pc = torch.cat([
                            one_pc[0] for one_pc in batch_proc_pc
                        ], dim=0).float().cuda()
                    except RuntimeError:
                        batch_proc_pc = [
                            one_pc[0][0] for one_pc in batch_proc_pc
                        ]  # list of [num, 3]

                    # get latent feature vector c
                    # c is [B, c_dim (typically 512)]
                    c = self.generator.model.encode_inputs(batch_proc_sel_pc)

                    # z is of no use
                    z = None

                # init points and optimize
                points = self.init_points(batch_proc_pc)
                points.requires_grad_()
                points = self.optimize_points(
                    points, 
                    z, 
                    c,
                    rep_weight=self.rep_weight,
                    iterations=self.iterations,
                    printing=False
                )
                opt_pc[idx:idx + self.batch_size] = points

        return torch.tensor(opt_pc).to(self.device)


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

