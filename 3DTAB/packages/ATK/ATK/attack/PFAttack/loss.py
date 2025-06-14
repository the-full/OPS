import torch
import torch.nn as nn
import torch.nn.functional as F


class PFLoss(nn.Module):
    def __init__(self, model, decomp_num=2):
        super(PFLoss, self).__init__()
        self.model      = model
        self.decomp_num = decomp_num

    def pert_factorize(self, pert):
        device = pert.device
        decomp_ids = torch.randint(0, self.decomp_num, size=pert.shape, device=device, dtype=torch.long)
        for d in range(self.decomp_num):
            yield (pert * decomp_ids.eq(d))

    def get_pf_loss(self, logits, labels, o_labels):
        _, N = logits.shape
        labels_one_hot = F.one_hot(labels, num_classes=N)
        others_one_hot = F.one_hot(o_labels, num_classes=N)
        g_logits = torch.sum(logits * labels_one_hot, dim=-1)
        o_logits = torch.sum(logits * others_one_hot, dim=-1)
        return (g_logits - o_logits) ** 2

    def get_logits(self, pcs, **kwargs):
        feat = kwargs.get('feat', pcs)
        return self.model.__predict__(dict(xyz = pcs, feat = feat))

    def forward(self, ori_pcs, adv_pcs, labels):
        main_logits = self.get_logits(adv_pcs)
        labels_one_hot = F.one_hot(labels, num_classes=main_logits.shape[-1])
        o_labels = torch.argmax(main_logits - labels_one_hot * 1e10, dim=1)

        main_pert = adv_pcs - ori_pcs
        pf_loss   = self.get_pf_loss(main_logits, labels, o_labels)
        for subs_pert in self.pert_factorize(main_pert):
            subs_adv_pcs = ori_pcs + subs_pert
            subs_logits  = self.get_logits(subs_adv_pcs)
            pf_loss = pf_loss + self.get_pf_loss(subs_logits, labels, o_labels)
        return pf_loss
