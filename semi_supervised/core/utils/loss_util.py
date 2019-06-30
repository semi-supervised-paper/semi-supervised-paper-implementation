import torch
import torch.nn.functional as F
from .fun_util import l2_normalize, kl_div, disable_tracking_bn_stats


class VATLoss(torch.nn.Module):

    def __init__(self, xi=1e-6, eps=8.0, iter_num=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param iter_num: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.iter_num = iter_num
    
    def norm_vector(self, v):
        v = v / (1e-12 + self.__reduce_max(v.abs(), range(1, len(v.shape))))
        v = v / (1e-6 + v.pow(2).sum((1, 2, 3), keepdim=True)).sqrt()
        return v

    def forward(self, model, x):
        # prepare random unit tensor
        d = torch.randn_like(x)
        d = self.norm_vector(d)
        #d /= (1e-12 + torch.max(torch.abs(d)))
        #d /= torch.sqrt(1e-12 + torch.sum(d ** 2))

        with disable_tracking_bn_stats(model):
            logits = model(x).detach()
            # calc adversarial direction
            for _ in range(self.iter_num):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                adv_distance = kl_div(logits, pred_hat)
                adv_distance.backward()
                d = d.grad
                #d /= (1e-12 + torch.max(torch.abs(d)))
                #d = d / torch.sqrt(1e-12 + torch.sum(d ** 2))
                d = self.norm_vector(d).detach()
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            lds = kl_div(pred_hat, logits)

        return lds

    def __reduce_max(self, v, idx_list):
        for i in idx_list:
            v = v.max(i, keepdim=True)[0]
        return v


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size(1)
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes
