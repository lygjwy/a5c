import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from .base_select import BaseSelect
from .utils import get_submodular_optimizer, FacilityLocation
from utils import euclidean_dist_pair_np
# from models import get_classifier


class CRAIG(BaseSelect):

    def __init__(self, num_classes, data_loader, models, greedy='lazy'):
        super().__init__(num_classes, data_loader)
        self.arch, self.pretrain = models[-1]
        self.greedy = greedy
        self.grads, self.labels = self.extract_grad_y()

    def extract_grad_y(self):
        gradients = []
        labels_all = torch.zeros([0], requires_grad=False).cuda()

        clf = get_classifier(self.arch, self.num_classes, self.pretrain)
        if torch.cuda.is_available():
            clf.cuda()
        cudnn.benchmark = True
        clf.eval()

        for data in self.data_loader:
            inputs, labels = data
            inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

            with torch.no_grad():
                z, h = clf(inputs, ret_feat=True)
            
            loss = F.cross_entropy(h.requires_grad_(True), labels).sum()
            bias_parameters_grads = torch.autograd.grad(loss, h)[0]
            weight_parameters_grads = z.view(len(labels), 1, clf.in_features).repeat(1, self.num_classes, 1) * bias_parameters_grads.view(len(labels), self.num_classes, 1).repeat(1, 1, clf.in_features)
            
            gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1).cpu().numpy())
            labels_all = torch.cat((labels_all, labels), dim=0)
        
        gradients = np.concatenate(gradients, axis=0)

        return euclidean_dist_pair_np(gradients), labels
    
    def calc_weights(self, matrix, result):
        min_sample = np.argmax(matrix[result], axis=0)
        weights = np.ones(np.sum(result) if result.dtype == bool else len(result))
        for i in min_sample:
            weights[i] = weights[i] + 1

        return weights
    
    def select_in_category(self, c, ratio):
        idxs_c = torch.where(self.labels == c)[0].cpu().numpy()
        num_c = len(idxs_c)
        num_selected_c = round(num_c * ratio)

        matrix = -1. * self.grads[idxs_c]
        matrix -= np.min(matrix) - 1e-3
        submod_function = FacilityLocation(idxs_c, similarity_matrix=matrix)
        submod_optimizer = get_submodular_optimizer(self.greedy, idxs_c, budget=num_selected_c)
        class_result = submod_optimizer.select(gain_function=submod_function.calc_gain, update_state=submod_function.update_state)
        # weights = self.calc_weights(matrix, np.isin(idxs_c, class_result))
        
        return class_result
