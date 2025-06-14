from .attack_template import IterAttack

class IEFGSM(IterAttack):
    ''' Euler's method I-FGSM (IE-FGSM).

    Ref: 
        - Boosting Transferability of Adversarial Example via an Enhanced Euler's Method (ICASSP 2023)
        - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096558
        - https://github.com/Trustworthy-AI-Group/TransferAttack/blob/main/transferattack/gradient/iefgsm.py
    '''
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)


    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta  = self.init_delta(ori_pcs)

        logits = self.get_logits(ori_pcs + delta)
        loss   = self.get_loss(logits, labels, target)
        grad_p = self.get_grad(loss, delta)
        g_p    = self.norm_grad(grad_p)

        logits = self.get_logits(ori_pcs + delta + self.alpha * g_p)
        loss   = self.get_loss(logits, labels, target)
        grad_a = self.get_grad(loss, delta)
        g_a    = self.norm_grad(grad_a)
        
        grad   = (g_a + g_p) / 2

        delta  = self.update_delta(delta, grad.sign())
        delta  = self.proj_delta(ori_pcs, delta, **kwargs)

        return delta.detach()
