from .attack_template import IterAttack


class IterateFGSM(IterAttack):
    ''' Iterative FGSM (I-FGSM).

    Ref: 
        - Adversarial Examples in the Physical World (ICLR 2017)
        - https://arxiv.org/pdf/1607.02533
    '''
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta = self.init_delta(ori_pcs)
        for _ in range(self.num_iter):
            logits = self.get_logits(ori_pcs + delta)
            loss   = self.get_loss(logits, labels, target)
            grad   = self.get_grad(loss, delta)
            delta  = self.update_delta(delta, grad.sign())
            delta  = self.proj_delta(ori_pcs, delta, **kwargs)
        return delta.detach()
