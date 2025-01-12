'''
Geometry-based selection algorithms: Herding, K-Center Greedy, and Contextual Diversity
'''

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from .base_select import BaseSelect
from utils import euclidean_dist
from models import get_cclf


class GeoSelect(BaseSelect):

    def __init__(self, num_classes, data_loader, models, metric):
        super().__init__(num_classes, data_loader)
        self.arch, self.ckpt = models[-1]
        self.zs, self.ys = self.extract_zs_ys()
        
        if metric == 'euclidean':
            self.metric = euclidean_dist
        else:
            raise ValueError('NOT SUPPORTED METRIC')
    
    def extract_zs_ys(self):
        cclf = get_cclf(self.arch, self.num_classes, self.ckpt)
        if torch.cuda.is_available():
            cclf.cuda()
        cudnn.benchmark = True
        cclf.eval()

        zs = None
        ys = torch.zeros([0], requires_grad=False, dtype=torch.long).cuda()

        for data in self.data_loader:
            inputs, labels = data
            inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

            with torch.no_grad():
                z = cclf.forward_features(inputs)
            
            if zs is None:
                zs = torch.flatten(F.adaptive_avg_pool2d(z, (1, 1)), start_dim=1)
            else:
                # z_all = torch.cat((z_all, z), dim=0)
                zs = torch.cat((zs, torch.flatten(F.adaptive_avg_pool2d(z, (1, 1)), start_dim=1)), dim=0)
            
            ys = torch.cat((ys, labels), dim=0)
        
        return zs, ys