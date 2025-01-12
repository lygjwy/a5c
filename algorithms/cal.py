import torch
import numpy as np

import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from .base_select import BaseSelect
from utils import euclidean_dist
from models import get_cclf

class Cal(BaseSelect):

    def __init__(self, num_classes, data_loader, models, metric='euclidean', neighbours=10):
        super().__init__(num_classes, data_loader)

        self.arch, self.ckpt = models[-1]
        if metric == 'euclidean':
            self.metric = euclidean_dist
        else:
            raise ValueError('NOT SUPPORTED METRIC')
        self.zs, self.hs, self.ys = self.extract_zs_hs_ys()
        self.neighbours = neighbours

    def extract_zs_hs_ys(self):
        cclf = get_cclf(self.arch, self.num_classes, self.ckpt)
        if torch.cuda.is_available():
            cclf.cuda()
        cudnn.benchmark = True
        cclf.eval()

        zs = None
        hs = torch.zeros([0, self.num_classes], requires_grad=False).cuda()
        ys = torch.zeros([0], requires_grad=False).cuda()

        for data in self.data_loader:
            inputs, labels = data
            inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

            with torch.no_grad():
                z = cclf.forward_features(inputs)
                logits = cclf(inputs)
                h = F.softmax(logits, dim=1)

            if zs is None:
                zs = torch.flatten(F.adaptive_avg_pool2d(z, (1, 1)), start_dim=1)
            else:
                zs = torch.cat((zs, torch.flatten(F.adaptive_avg_pool2d(z, (1, 1)), start_dim=1)), dim=0)
            
            hs = torch.cat((hs, h), dim=0)
            ys = torch.cat((ys, labels), dim=0)

        return zs, hs, ys

    def select_in_category(self, c, ratio):
        
        idxs_c = torch.where(self.ys == c)[0]
        num_c = len(idxs_c)
        num_selected_c = round(num_c * ratio)

        z_c = self.zs[idxs_c]
        h_c = self.hs[idxs_c]
        
        knn = np.argsort(self.metric(z_c, z_c), axis=1)[:, 1:(self.neighbors + 1)]
        
        aa = np.expand_dims(h_c, 1).repeat(self.neighbors, 1)
        bb = h_c[knn, :]
        scores_c = np.mean(np.sum(0.5 * aa * np.log(aa / bb) + 0.5 * bb * np.log(bb / aa), axis=2), axis=1)
        
        return idxs_c[np.argsort(scores_c)[::1][:num_selected_c]]
