from .attack_template import IterAttack

class FGSM(IterAttack):
    ''' Fast Gradient Sign Method (FGSM). 

    Ref: 
        - https://arxiv.org/pdf/1412.6572 .
    '''

    def __init__(self, model, **kwargs):
        super().__init__(model, num_iter=1, **kwargs)

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta  = self.init_delta(ori_pcs)
        logits = self.get_logits(ori_pcs + delta)
        loss   = self.get_loss(logits, labels, target)
        grad   = self.get_grad(loss, delta)
        delta  = self.update_delta(delta, grad.sign())
        delta  = self.proj_delta(ori_pcs, delta, **kwargs)
        return delta.detach()

