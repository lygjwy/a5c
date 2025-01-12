'''
GraNd score: gradient L2 norm
'''
import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.func import functional_call, vmap, grad
import torch.nn.functional as F

from .base_select import BaseSelect
from models import get_cclf


class GraNd(BaseSelect):

    def __init__(self, num_classes, data_loader, models):
        super().__init__(num_classes, data_loader)

        self.arch, self.ckpt = models[-1]
        self.scores, self.labels = self.extract_grand_y()

    def extract_grand_y(self):

        scores = torch.zeros([0], requires_grad=False).cuda()
        labels_all = torch.zeros([0], requires_grad=False).cuda()

        cclf = get_cclf(self.arch, self.num_classes, self.ckpt)
        if torch.cuda.is_available():
            cclf.cuda()
        cudnn.benchmark = True

        params = {k: v.detach() for k, v in cclf.named_parameters()}
        buffers = {k: v.detach() for k, v in cclf.named_buffers()}
        cclf.eval()
        
        for data in self.data_loader:

            inputs, labels = data
            inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

            # calculate gradients norm
            def compute_loss(params, buffers, sample, target):
                batch = sample.unsqueeze(0)
                targets = target.unsqueeze(0)

                predictions = functional_call(cclf, (params, buffers), (batch,))
                loss = F.cross_entropy(predictions, targets)
                
                return loss

            ft_per_sample_grads = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))(params, buffers, inputs, labels)
            # flatten, if accumulate?
            sample_grads = torch.zeros([len(labels), 0]).cuda()
            for name in ft_per_sample_grads.keys():
                sample_grads = torch.hstack((sample_grads, torch.flatten(ft_per_sample_grads[name], start_dim=1)))            
            
            scores = torch.cat((scores, torch.norm(sample_grads, dim=1)), dim=0)
            
            labels_all = torch.cat((labels_all, labels), dim=0)

        return scores.cpu().numpy(), labels_all
    

    def select_in_category(self, c, ratio):

        idxs_c = torch.where(self.labels == c)[0].cpu().numpy()
        num_c = len(idxs_c)

        scores_c = self.scores[idxs_c]
        num_selected_c = round(num_c * ratio)

        return idxs_c[np.argsort(scores_c)[::-1][:num_selected_c]]
