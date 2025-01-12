'''
Uncertainty based selection algorithms: LeastConfidence, Entropy, Margin
'''
import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from .base_select import BaseSelect
from models import get_cclf


class Uncertainty(BaseSelect):

    def __init__(self, num_classes, data_loader, models, principle):
        super().__init__(num_classes, data_loader)

        # parse
        # self.models = models
        self.arch, self.ckpt = models[-1]
        self.principle = principle
        self.scores, self.labels = self.extract_uncertainty_y()


    def extract_uncertainty_y(self):

        scores = np.array([])
        labels_all = torch.zeros([0], requires_grad=False).cuda()
        
        cclf = get_cclf(self.arch, self.num_classes, self.ckpt)
        if torch.cuda.is_available():
            cclf.cuda()
        cudnn.benchmark = True
        cclf.eval()
        
        for data in self.data_loader:

            inputs, labels = data
            inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

            with torch.no_grad():
                logits = cclf(inputs)
                preds = F.softmax(logits, dim=1)
            
            if self.principle == 'leastconfidence':
                # preds = F.softmax(logits, dim=1).cpu().numpy()
                scores = np.append(scores, np.max(preds.cpu().numpy(), axis=1))
            elif self.principle == 'entropy':
                # preds = F.softmax(preds, dim=1).cpu().numpy()
                scores = np.append(scores, (np.log(preds.cpu().numpy() + 1e-10) * preds.cpu().numpy()).sum(axis=1))
            elif self.principle == 'margin':
                preds_argmax = torch.argmax(preds, dim=1)
                max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                preds_sub_argmax = torch.argmax(preds, dim=1)
                scores = np.append(scores, (max_preds - preds[torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy())
            else:
                raise ValueError('NOT SUPPORTED PRINCIPLE')
            
            labels_all = torch.cat((labels_all, labels), dim=0)

        return scores, labels_all


    def select_in_category(self, c, ratio):
        idxs_c = torch.where(self.labels == c)[0].cpu().numpy()
        num_c = len(idxs_c)

        scores_c = self.scores[idxs_c]
        budget = round(num_c * ratio)
        
        return idxs_c[np.argsort(scores_c)[::-1][:budget]]
