import torch
import numpy as np

from ATK.utils.advloss import CWAdvLoss

from ..attack_template import IterAttack


class SIFGSM(IterAttack):
    ''' Shape Invariant FGSM (SI-FGSM). 

    Ref: 
    '''

    def __init__(self, model, **kwargs):
        kwargs.setdefault('loss_fn', CWAdvLoss(kappa=0.))
        kwargs.setdefault('loss_type', 'cw')
        super().__init__(model, num_iter=1, **kwargs)


    def get_loss(self, logits, labels, target=None): 
        if self.loss_type == 'cw':
            return self.loss_fn(logits, labels, target=target).mean()
        else:
            super().get_loss(logits, labels, target=target)


    def attack(self, ori_pcs, labels, target=None, **kwargs):
        _, N, C = ori_pcs.shape
        normal_vec = kwargs.get('normal', None)
        if normal_vec is None:
            normal_vec = self.estimate_pcs_normals(ori_pcs) # <B, N, 3>
        adv_pcs = ori_pcs.clone()
        
        for _ in range(self.num_iter):
            # P -> P', detach()
            new_adv_pcs, spin_axis_matrix, translation_matrix = self.get_transformed_point_cloud(ori_pcs, normal_vec)
            new_adv_pcs = new_adv_pcs.detach()
            new_adv_pcs.requires_grad = True

            # P' -> P
            adv_pcs = self.get_original_point_cloud(new_adv_pcs, spin_axis_matrix, translation_matrix)

            # get white-box gradients
            logits = self.get_logits(adv_pcs)
            loss   = self.get_loss(logits, labels, target=target)
            grad   = self.get_grad(loss, new_adv_pcs)
            grad[:, :, 2] = 0.

            # update P', P and N
            norm = grad.norm(dim=(1, 2), keepdim=True)
            normed_grad = grad / (norm + 1e-9)

            new_adv_pcs = new_adv_pcs - self.alpha * np.sqrt(N*C) * normed_grad
            adv_pcs = self.get_original_point_cloud(new_adv_pcs, spin_axis_matrix, translation_matrix)

            delta = adv_pcs - ori_pcs
            delta = self.proj_delta(ori_pcs, delta)
            adv_pcs = ori_pcs + delta

            normal_vec = self.estimate_pcs_normals(adv_pcs)

        delta = adv_pcs - ori_pcs
        return delta.detach()


    def get_spin_axis_matrix(self, normal_vec):
        """Calculate the spin-axis matrix.

        Args:
            normal_vec (torch.cuda.FloatTensor<B, N, 3>): the normal vectors for all N points, [1, N, 3].
        """
        B, N, _ = normal_vec.shape
        x = normal_vec[:,:,0] # [1, N]
        y = normal_vec[:,:,1] # [1, N]
        z = normal_vec[:,:,2] # [1, N]
        assert abs(normal_vec).max() <= 1
        u = torch.zeros(B, N, 3, 3).cuda()
        denominator = torch.sqrt(1-z**2) # \sqrt{1-z^2}, [1, N]
        u[:,:,0,0] = y / denominator
        u[:,:,0,1] = - x / denominator
        u[:,:,0,2] = 0.
        u[:,:,1,0] = x * z / denominator
        u[:,:,1,1] = y * z / denominator
        u[:,:,1,2] = - denominator
        u[:,:,2] = normal_vec
        # revision for |z| = 1, boundary case.
        pos = torch.where(abs(z ** 2 - 1) < 1e-4)
        u[pos[0], pos[1], 0, 0] = 1 / np.sqrt(2)
        u[pos[0], pos[1], 0, 1] = - 1 / np.sqrt(2)
        u[pos[0], pos[1], 0, 2] = 0.
        u[pos[0], pos[1], 1, 0] = z[pos] / np.sqrt(2)
        u[pos[0], pos[1], 1, 1] = z[pos] / np.sqrt(2)
        u[pos[0], pos[1], 1, 2] = 0.
        u[pos[0], pos[1], 2, 0] = 0.
        u[pos[0], pos[1], 2, 1] = 0.
        u[pos[0], pos[1], 2, 2] = z[pos]
        return u.data


    def get_transformed_point_cloud(self, points, normal_vec):
        """Calculate the spin-axis matrix.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 3].
            normal_vec (torch.cuda.FloatTensor): the normal vectors for all N points, [1, N, 3].
        """
        intercept = torch.mul(points, normal_vec).sum(-1, keepdim=True) # P \cdot N, [1, N, 1]
        spin_axis_matrix = self.get_spin_axis_matrix(normal_vec) # U, [1, N, 3, 3]
        translation_matrix = torch.mul(intercept, normal_vec).data # (P \cdot N) N, [1, N, 3]
        new_points = points + translation_matrix #  P + (P \cdot N) N, [1, N, 3]
        new_points = new_points.unsqueeze(-1) # P + (P \cdot N) N, [1, N, 3, 1]
        new_points = torch.matmul(spin_axis_matrix, new_points) # P' = U (P + (P \cdot N) N), [1, N, 3, 1]
        new_points = new_points.squeeze(-1).data # P', [1, N, 3]
        return new_points, spin_axis_matrix, translation_matrix


    def get_original_point_cloud(self, new_points, spin_axis_matrix, translation_matrix):
        """Calculate the spin-axis matrix.

        Args:
            new_points (torch.cuda.FloatTensor): the transformed point cloud with N points, [1, N, 3].
            spin_axis_matrix (torch.cuda.FloatTensor): the rotate matrix for transformation, [1, N, 3, 3].
            translation_matrix (torch.cuda.FloatTensor): the offset matrix for transformation, [1, N, 3, 3].
        """
        inputs = torch.matmul(spin_axis_matrix.transpose(-1, -2), new_points.unsqueeze(-1)) # U^T P', [1, N, 3, 1]
        inputs = inputs - translation_matrix.unsqueeze(-1) # P = U^T P' - (P \cdot N) N, [1, N, 3, 1]
        inputs = inputs.squeeze(-1) # P, [1, N, 3]
        return inputs

