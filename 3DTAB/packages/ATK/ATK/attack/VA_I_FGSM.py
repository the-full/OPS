import torch
import numpy as np

from .attack_template import IterAttack

class VAIFGSM(IterAttack):
    ''' Momentum Iterative FGSM (MI-FGSM).

    Ref: 
        - mproving Transferability of Adversarial Examples with Virtual Step and Auxiliary Gradients (IJCAI 2022)
        - https://www.ijcai.org/proceedings/2022/0227.pdf
        - https://github.com/Trustworthy-AI-Group/TransferAttack/blob/main/transferattack/gradient/vaifgsm.py
    '''
    def __init__(
        self, 
        model, 
        num_iter = 20,
        alpha = 0.007,
        num_aux = 3,
        num_classes = 40,
        **kwargs
    ):
        super().__init__(
            model, 
            num_iter = num_iter,
            alpha = alpha, 
            **kwargs
        )
        self.num_classes = num_classes
        self.num_aux     = num_aux


    def set_budget(
        self, 
        budget: float = 0.45,
        budget_type: str = 'linfty',
        update_alpha: bool = True,
    ):
        super().set_budget(budget, budget_type, False)


    def get_aux_labels(self, labels):   
        """
        Generate auxiliary label.
        """
        aux_labels = []
        for i in range(labels.shape[0]):
            aux_label = torch.randperm(self.num_classes).tolist()
            aux_label.remove(labels[i].item())
            aux_label = aux_label[:self.num_aux]
            aux_labels.append(aux_label)

        # Reshape from [batch_size, aux_num] to [aux_num, batch_size]
        aux_labels = np.transpose(np.array(aux_labels, dtype=np.int64),(1,0))
        
        aux_labels_list = []
        for i in range(aux_labels.shape[0]):
            aux_labels_list.append(torch.from_numpy(aux_labels[i]).detach().to(self.device))

        return aux_labels_list # [<B,>]

    def get_grad(self, loss, data, norm=None):
        grad = torch.autograd.grad(loss, data, retain_graph=True, create_graph=False)[0]
        return grad

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta = self.init_delta(ori_pcs)
        for _ in range(self.num_iter):
            logits = self.get_logits(ori_pcs + delta)
            loss   = self.get_loss(logits, labels, target)
            grad   = self.get_grad(loss, delta)
            delta  = self.update_delta(delta, grad.sign())

            aux_labels_list = self.get_aux_labels(labels)
            for aux_labels in aux_labels_list:
                aux_loss = self.get_loss(logits, aux_labels)
                aux_grad = self.get_grad(aux_loss, delta)
                delta    = self.update_delta(delta, -aux_grad.sign())

            logits.detach()

        delta = self.proj_delta(ori_pcs, delta, **kwargs)

        return delta.detach()
