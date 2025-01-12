import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from .base_select import BaseSelect
# from models import get_classifier


class Maha(BaseSelect):

    def __init__(self, num_classes, data_loader, ratio, models, principle):
        super().__init__(num_classes, data_loader, ratio)

        self.arch, self.pretrain = models[-1]
        self.principle = principle
        self.z_all, self.y_all = self.extract_z_y()
        self.cov = torch.cov(self.z_all.T)

    def get_mahas(self, idxs):
        z = self.z_all[idxs]
        u = torch.mean(z, dim=0)
        s = []

        for z_ in z:
            delta = z_ - u
            m = torch.dot(delta, torch.matmul(torch.inverse(self.cov), delta))
            s.append(torch.sqrt(m).item())

        return s

    def extract_z_y(self):
        clf = get_classifier(self.arch, self.num_classes, self.pretrain)
        if torch.cuda.is_available():
            clf.cuda()
        cudnn.benchmark = True
        clf.eval()
        self.in_features = clf.in_features

        z_all = torch.zeros([0, self.in_features], requires_grad=False).cuda()
        y_all = torch.zeros([0], requires_grad=False).cuda()
        
        for data in self.data_loader:
            inputs, labels = data
            inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())
            
            with torch.no_grad():
                z, _ = clf(inputs, ret_feat=True)

            z_all = torch.cat((z_all, z), dim=0)
            y_all = torch.cat((y_all, labels), dim=0)

        return z_all, y_all

    def select_in_category(self, c):
        idxs_c = torch.where(self.y_all == c)[0]
        budget = round(len(idxs_c) * self.ratio)
        scores_c = self.get_mahas(idxs_c)

        idxs_c = idxs_c.cpu().numpy()
        if self.principle == 'maha_a':
            return idxs_c[np.argsort(scores_c)[::-1][:budget]]
        elif self.principle == 'maha_t':
            return idxs_c[np.argsort(scores_c)[::1][:budget]]
        else:
            raise ValueError('ILLEGAL PRINCIPLE')