import torch
import torch.optim as optim
from einops import rearrange
from pytorch3d.ops import knn_points, knn_gather
from omegaconf import DictConfig

from ATK.utils.ops import find_offset, offset_proj
from ATK.utils.ops import farthest_point_sample
from ATK.utils.common import check_option
from ATK.utils.advloss import (
    NegtiveAdvLoss,
    CWAdvLoss,
    ChamferLoss,
    L2NormLoss,
    HausdorffLoss,
    CurvatureLoss,
    UniformLossGeoA3,
    NoneLoss,
)

from ..attack_template import CWAttack


class GeoA3(CWAttack):
    ''' 
    Ref:
        - Geometry-Aware Generation of Adversarial Point Clouds (TPAMI 2020)
        - https://arxiv.org/abs/1912.11171
        - https://github.com/Gorilla-Lab-SCUT/GeoA3/tree/master
    '''

    _attack_name = 'GeoA3'
    
    def __init__(
        self, 
        model, 
        optim_type        = 'Adam',
        adv_loss_type     = 'CE',
        kappa             = 0.,
        dis_loss_type     = 'CD',
        dis_loss_weight   = 1.0,
        hau_loss_weight   = 0.1,
        cur_loss_weight   = 1.0,
        knn_k_for_curve   = 16,
        uni_loss_weight   = 0.0,
        use_lr_scheduler  = False,
        max_atk_points    = 1024,
        is_subsample_opt  = False,
        eval_num          = 1,

        proj_cfg = DictConfig(dict(
            norm_proj = False,
            use_real  = False,
            lp_clip   = False,
        )),

        jitter_cfg = DictConfig(dict(
            open     = False,
            interval = 10,
            k        = 16,
            sigma    = 0.01,
            clip     = 0.05,
        )),
        **kwargs,
    ):
        kwargs.setdefault('attack_lr', 0.01)
        kwargs.setdefault('initial_weight', 10.)
        kwargs.setdefault('binary_update_cond', 'success at last')
        super().__init__(model, **kwargs)

        check_option(adv_loss_type, ['CE', 'CW', 'none'])
        check_option(dis_loss_type, ['CD', 'L2', 'none'])
        check_option(optim_type, ['Adam', 'SGD'])

        self.adv_loss_fn.add_objective(
            'adv_loss',
            self.parse_adv_loss_fn(adv_loss_type, kappa=kappa),
            weight=1.0,
        )

        self.res_loss_fn.add_objective(
            'dis_loss',
            self.parse_dis_loss_fn(dis_loss_type),
            dis_loss_weight,
        )
        self.res_loss_fn.add_objective(
            'hau_loss',
            HausdorffLoss(single=True),
            hau_loss_weight,
        )
        self.res_loss_fn.add_objective(
            'cur_loss',
            CurvatureLoss(k=knn_k_for_curve),
            cur_loss_weight,
        )
        self.res_loss_fn.add_objective(
            'uni_loss',
            UniformLossGeoA3(),
            uni_loss_weight,
        )

        self.optim            = optim_type
        self.use_scheduler    = use_lr_scheduler
        self.knn_k_for_curve  = knn_k_for_curve
        self.is_subsample_opt = is_subsample_opt
        self.max_atk_points   = max_atk_points
        self.eval_num         = eval_num

        self.jitter_cfg       = DictConfig(jitter_cfg)
        self.proj_cfg         = DictConfig(proj_cfg)


    class FakeScheduler:
        def __init__(self, *args, **kwargs):
            pass

        def step(self):
            pass

    @staticmethod
    def parse_adv_loss_fn(adv_loss_type, **kwargs):
        adv_loss_map = {
            'CE':   NegtiveAdvLoss,
            'CW':   CWAdvLoss,
            'none': NoneLoss,
        }
        Loss = adv_loss_map[adv_loss_type]
        if adv_loss_type == 'CW':
            kappa =kwargs.get('kappa', 0.)
            return Loss(kappa)
        else:
            return Loss()

    @staticmethod
    def parse_dis_loss_fn(dis_loss_type, **kwargs):
        dis_loss_map = {
            'CD':  ChamferLoss,
            'L2':  L2NormLoss,
            'none': NoneLoss,
        }
        Loss = dis_loss_map[dis_loss_type]
        if dis_loss_type == 'CD':
            single = kwargs.get('single', False)
            reduce = kwargs.get('reduce', 'sum')
            return Loss(single, reduce)
        else:
            return Loss()
    

    def binary_search_coef(self):
        _c = self.coef_bound
        attack_success = self.outer_attack_achieve()

        B = _c.c_for_each_pc.shape[0]
        for k in range(B):
            if attack_success[k]:
                _c.c_lower_bound[k] = max(_c.c_for_each_pc[k], _c.c_lower_bound[k])
                if _c.c_upper_bound[k] < 1e9:
                    _c.c_for_each_pc[k] = (_c.c_lower_bound[k] + _c.c_upper_bound[k]) * 0.5
                else:
                    _c.c_for_each_pc[k] *= 2
            else:
                _c.c_upper_bound[k] = min(_c.c_for_each_pc[k], _c.c_upper_bound[k])
                if _c.c_upper_bound[k] < 1e9:
                    _c.c_for_each_pc[k] = (_c.c_lower_bound[k] + _c.c_upper_bound[k]) * 0.5


    def set_pbar_info_inner(
        self, 
        adv_info, 
        res_info, 
        **kwargs
    ):
        super().set_pbar_info_inner(adv_info, res_info)
        if res_info is not None:
            self.pbar.set_infos({
                'dis_loss': f'{res_info.dis_loss.mean().item():.4f}',
                'hau_loss': f'{res_info.hau_loss.mean().item():.4f}',
                'cur_loss': f'{res_info.cur_loss.mean().item():.4f}',
                'uni_loss': f'{res_info.uni_loss.mean().item():.4f}',
            })

    @torch.no_grad()
    def update_record_items(
        self, 
        adv_pcs, 
        dist_val, 
        labels, 
        target=None,
        **kwargs,
    ):
        _g = self.global_val
        B  = adv_pcs.shape[0]

        if _g.use_subsample:
            full_adv_pcs   = _g.full_adv_pcs
            attack_success = torch.full((B,), False).to(self.device)
            for k in range(B):
                eval_pcs = farthest_point_sample(
                    rearrange(full_adv_pcs[k], 'N C -> B N C', B=self.eval_num),
                    k = self.max_atk_points,
                )
                eval_logits = self.get_logits(eval_pcs)
                eval_result = self.attack_achieved(
                    logits = eval_logits, 
                    labels = labels[k].repeat(self.eval_num), 
                    target = target,
                ).sum()
                if eval_result > 0.5 * self.eval_num:
                    attack_success[k] = True
        else:
            predict_logits = self.get_logits(adv_pcs) # <B, num_class>
            attack_success = self.attack_achieved(predict_logits, labels, target) # <B,>

        super().update_record_items(adv_pcs, dist_val, labels, target, attack_success=attack_success)


    def inner_loop(self, ori_pcs, labels, target=None, **kwargs):
        B, N, _ = ori_pcs.shape
        device  = self.device
        _g = self.global_val

        _g.use_subsample = (
            (N > self.max_atk_points) and self.is_subsample_opt
        )
        constrain_loss = torch.full((B,), 1e10).to(device)
        c_for_each_pc = self.coef_bound.c_for_each_pc

        offset = torch.zeros_like(ori_pcs).normal_(mean=0, std=1e-3) # <B, N, C>
        offset.requires_grad_(True)

        # optimizer
        if self.optim == 'Adam':
            optimizer = optim.Adam([offset], lr=self.lr)
        elif self.optim == 'SGD':
            optimizer = optim.SGD([offset], lr=self.lr)
        else:
            assert False, 'Not support such optimizer.'

        # scheduler
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.9990, last_epoch=-1
            )
        else:
            scheduler = self.FakeScheduler()

        for step in self.pbar:
            adv_pcs = ori_pcs + offset # <B, N, C>

            if _g.use_subsample:
                _g.full_adv_pcs = adv_pcs.clone().detach()
                adv_pcs = farthest_point_sample(adv_pcs, self.max_atk_points)[0] # <B, N', C> N'=num_atk_points

            self.update_record_items(adv_pcs, constrain_loss, labels, target=target)

            adv_pcs  = self.pre_jitter(adv_pcs, step)
            logits   = self.get_logits(adv_pcs) # <B, num_class>
            adv_info = self.adv_loss_fn(
                adv_loss = (logits, labels, target),
            ) # <B,>
            # NOTE: https://github.com/Gorilla-Lab-SCUT/GeoA3/blob/master/Lib/loss_utils.py#L45-L50 
            res_info = self.res_loss_fn(
                dis_loss = (ori_pcs, adv_pcs),
                hau_loss = (adv_pcs, ori_pcs),
                cur_loss = (ori_pcs, adv_pcs, kwargs['normal']),
                uni_loss = (adv_pcs),
            ) # <B,>

            constrain_loss = res_info.loss

            adv_loss, res_loss = adv_info.loss, res_info.loss
            loss = torch.mean(adv_loss + c_for_each_pc * res_loss) # <1,>

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            self.set_pbar_info_inner(adv_info, res_info)

            ori_normals = kwargs.get('normal', None)
            offset.data = self.proj_offset(
                ori_pcs, adv_pcs, offset.data, ori_normals, **kwargs
            )


    def attack(self, ori_pcs, labels, target=None, **kwargs):
        ori_normals = kwargs.get('normal', None)
        if ori_normals is None:
            ori_normals = self.estimate_pcs_normals(ori_pcs)
            kwargs['normal'] = ori_normals

        delta = super().attack(ori_pcs, labels, target=target, **kwargs)
        return delta


    def pre_jitter(self, adv_pcs, step):
        if self.jitter_cfg.open:
            _g = self.global_val

            if step % self.jitter_cfg.interval == 0:
                project_noise = self.estimate_perpendicular(adv_pcs)
                _g.project_noise = project_noise
            else:
                project_noise = _g.project_noise

            adv_pcs.data = adv_pcs.data + project_noise

        return adv_pcs


    def proj_offset(
        self, 
        ori_pcs, 
        adv_pcs, 
        offset, 
        ori_normals=None, 
        **kwargs
    ):
        if self.proj_cfg.norm_proj:
            if self.proj_cfg.use_real:
                offset = find_offset(ori_pcs, adv_pcs)
            assert ori_normals is not None
            offset = offset_proj(offset, ori_pcs, ori_normals)

        if self.proj_cfg.lp_clip:
            offset = self.proj_delta(ori_pcs, offset, **kwargs)

        return offset


    def estimate_perpendicular(self, pcs):
        pcs = pcs.permute(0, 2, 1)
        k     = self.jitter_cfg.k
        sigma = self.jitter_cfg.sigma
        clip  = self.jitter_cfg.clip

        with torch.no_grad():
            # pc : [b, 3, n]
            b,_,n=pcs.size()
            inter_KNN = knn_points(pcs.permute(0,2,1), pcs.permute(0,2,1), K=k+1) #[dists:[b,n,k+1], idx:[b,n,k+1]]
            nn_pts = knn_gather(pcs.permute(0,2,1), inter_KNN.idx).permute(0,3,1,2)[:,:,:,1:].contiguous() # [b, 3, n ,k]

            # get covariance matrix and smallest eig-vector of individual point
            perpendi_vector_1 = []
            perpendi_vector_2 = []
            for i in range(b):
                curr_point_set = nn_pts[i].detach().permute(1,0,2) #curr_point_set:[n, 3, k]
                curr_point_set_mean = torch.mean(curr_point_set, dim=2, keepdim=True) #curr_point_set_mean:[n, 3, 1]
                curr_point_set = curr_point_set - curr_point_set_mean #curr_point_set:[n, 3, k]
                curr_point_set_t = curr_point_set.permute(0,2,1) #curr_point_set_t:[n, k, 3]
                fact = 1.0 / (k-1)
                cov_mat = fact * torch.bmm(curr_point_set, curr_point_set_t) #curr_point_set_t:[n, 3, 3]
                eigenvalue, eigenvector=torch._linalg_utils._symeig(cov_mat, eigenvectors=True)    # eigenvalue:[n, 3], eigenvector:[n, 3, 3]

                larger_dim_idx = torch.topk(eigenvalue, 2, dim=1, largest=True, sorted=False)[1] # eigenvalue:[n, 2]

                persample_perpendi_vector_1 = torch.gather(eigenvector, 2, larger_dim_idx[:,0].unsqueeze(1).unsqueeze(2).expand(n, 3, 1)).squeeze() #persample_perpendi_vector_1:[n, 3]
                persample_perpendi_vector_2 = torch.gather(eigenvector, 2, larger_dim_idx[:,1].unsqueeze(1).unsqueeze(2).expand(n, 3, 1)).squeeze() #persample_perpendi_vector_2:[n, 3]

                perpendi_vector_1.append(persample_perpendi_vector_1.permute(1,0))
                perpendi_vector_2.append(persample_perpendi_vector_2.permute(1,0))

            perpendi_vector_1 = torch.stack(perpendi_vector_1, 0) #perpendi_vector_1:[b, 3, n]
            perpendi_vector_2 = torch.stack(perpendi_vector_2, 0) #perpendi_vector_1:[b, 3, n]

            aux_vector1 = sigma * torch.randn(b,n).unsqueeze(1).cuda() #aux_vector1:[b, 1, n]
            aux_vector2 = sigma * torch.randn(b,n).unsqueeze(1).cuda() #aux_vector2:[b, 1, n]

        return torch.clamp(perpendi_vector_1*aux_vector1, -1*clip, clip) + torch.clamp(perpendi_vector_2*aux_vector2, -1*clip, clip)
