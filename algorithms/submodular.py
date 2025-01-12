import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from .base_select import BaseSelect
from .utils import get_submodular_function, get_submodular_optimizer
from utils import cossim_np
# from models import get_classifier


class Submodular(BaseSelect):

    def __init__(self, num_classes, data_loader, ratio, models, function='logdeterminant', greedy='approximate', metirc='cossim'):
        super().__init(num_classes, data_loader, ratio)
        self.arch, self.pretrain = models[-1]
        self.function = function
        self.greedy = greedy
        self.gradients, self.labels = self.extract_grad_y()

    def extract_grad_y(self):
        gradients = []
        labels_all = torch.zeros([0], requires_grad=False).cuda()
        
        clf = get_classifier(self.arch, self.num_classes, self.pretrain)
        if torch.cuda.is_available():
            clf.cuda()
        cudnn.benchmark = True
        clf.eval()
        self.in_features = clf.in_features

        for data in self.data_loader:
            inputs, labels = data
            inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

            # with torch.no_grad():
            z, h = clf(inputs, ret_feat=True)
            loss = F.cross_entropy(h.requires_grad_(True), labels).sum()
            batch_num = len(labels)
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, h)[0]
                weight_parameters_grads = z.view(batch_num, 1, self.in_features).repeat(1, self.num_classes, 1) * bias_parameters_grads.view(batch_num, self.num_classes, 1).repeat(1, 1, self.in_features)

                gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1).cpu().numpy())
            
            labels_all = torch.cat((labels_all, labels), dim=0)
        gradients = np.concatenate(gradients, axis=0)

        return gradients, labels_all

    def select_in_category(self, c):
        idxs_c = torch.where(self.labels == c)[0].cpu().numpy()
        num_c = len(idxs_c)
        num_selected_c = round(num_c * self.ratio)

        grads_c = self.gradients[idxs_c]
        submod_func = get_submodular_function(self.function, idxs_c, similarity_kernel=lambda a, b:cossim_np(grads_c[a], grads_c[b]))
        submod_optimizer = get_submodular_optimizer(self.greedy, idxs_c, num_selected_c)

        return submod_optimizer.select(submod_func.calc_gain, update_state=submod_func.update_state)


