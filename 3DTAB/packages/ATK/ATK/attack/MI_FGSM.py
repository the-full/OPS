import torch

from .attack_template import IterAttack

class MIFGSM(IterAttack):
    ''' Momentum Iterative FGSM (MI-FGSM).

    Ref: 
        - Boosting Adversarial Attacks with Momentum (CVPR 2018)
          ! NIPS 2017 Competition Track Rank 1 in Untarget/Targeted Attack
            (See https://nips.cc/Conferences/2017/CompetitionTrack)
        - https://arxiv.org/abs/1710.06081
        - https://github.com/Jeffkang-94/pytorch-adversarial-attack/blob/master/attack/mifgsm.py
        - https://github.com/thu-ml/ares/blob/main/ares/attack/mim.py
        - https://github.com/dongyp13/Non-Targeted-Adversarial-Attacks/blob/master/attack_iter.py

    '''
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta    = self.init_delta(ori_pcs)
        momentum = torch.zeros_like(ori_pcs)

        for _ in range(self.num_iter):
            logits   = self.get_logits(ori_pcs + delta)
            loss     = self.get_loss(logits, labels, target)
            grad     = self.get_grad(loss, delta)
            momentum = self.update_momentum(momentum, grad)
            delta    = self.update_delta(delta, momentum.sign())
            delta    = self.proj_delta(ori_pcs, delta, **kwargs)

        return delta.detach()
