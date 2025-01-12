import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F

from .base_select import BaseSelect
from models import get_cclf


class CumuLoss(BaseSelect):

    def __init__(self, num_classes, data_loader, models, mode="cle"):
        super().__init__(num_classes, data_loader)
        self.models = models
        self.mode = mode
        self.cl, self.labels = self.extract_cl_y()

    def extract_cl_y(self):

        # record loss from different epoch clf
        loss_all = []
        for i in range(len(self.models) - 1):
            loss_all.append(np.array([]))

        for i, model in enumerate(self.models[1:]):
            arch, ckpt = model
            cclf = get_cclf(arch, self.num_classes, ckpt)
            if torch.cuda.is_available():
                cclf.cuda()
            cudnn.benchmark = True

            cclf.eval()

            labels_all = torch.zeros([0], requires_grad=False).cuda()
            for data in self.data_loader:
                inputs, labels = data
                inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

                with torch.no_grad():
                    logits = cclf(inputs)
                    ce = F.cross_entropy(logits, labels, reduction='none')
                    loss_all[i] = np.append(loss_all[i], ce.cpu().numpy())
            
            labels_all = torch.cat((labels_all, labels), dim=0)

        # calculate cumulative loss
        cumu_loss = np.average(np.array(loss_all), axis=0)

        return cumu_loss, labels_all

    def select_in_category(self, c, r):
        
        idxs_c = torch.where(self.labels == c)[0].cpu().numpy()
        num_c = len(idxs_c)

        scores_c = self.cl[idxs_c]
        num_selected_c = round(num_c * r)

        if self.mode == 'cle':
            return idxs_c[np.argsort(scores_c)[:num_selected_c]]
        else:
            return idxs_c[np.argsort(scores_c)[::-1][:num_selected_c]]