'''
GraNd score: gradient L2 norm
'''
import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from .base_select import BaseSelect
from models import get_cclf


class EL2N(BaseSelect):

    def __init__(self, num_classes, data_loader, models):
        super().__init__(num_classes, data_loader)

        self.arch, self.ckpt = models[-1]
        self.scores, self.labels = self.extract_el2n_y()

    def extract_el2n_y(self):

        scores = torch.zeros([0], requires_grad=False).cuda()
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

            scores = torch.cat((scores, torch.norm(preds-F.one_hot(labels, num_classes=self.num_classes), dim=1)), dim=0)
            labels_all = torch.cat((labels_all, labels), dim=0)

        return scores.cpu().numpy(), labels_all
    

    def select_in_category(self, c, ratio):

        idxs_c = torch.where(self.labels == c)[0].cpu().numpy()
        num_c = len(idxs_c)

        scores_c = self.scores[idxs_c]
        num_selected_c = round(num_c * ratio)

        return idxs_c[np.argsort(scores_c)[::-1][:num_selected_c]]
