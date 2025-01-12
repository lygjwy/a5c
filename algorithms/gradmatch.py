import numpy as np
from scipy.optimize import nnls

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from .base_select import BaseSelect
# from models import get_classifier


class GradMatch(BaseSelect):

    def __init__(self, num_classes, data_loader, ratio, models, val=False, lam=1.0):
        super().__init__(num_classes, data_loader, ratio)

        self.arch, self.pretrain = models[-1]
        self.val = val
        self.lam = lam
        self.grads, self.labels = self.extract_grad_y()

    def orthogonal_matching_pursuit(self, A, b, budget):
        with torch.no_grad():
            d, n = A.shape
            if budget <= 0:
                budget = 0
            elif budget > n:
                budget = n
            
            x = np.zeros(n, dtype=np.float32)
            resid = b.clone()
            indices = []
            boolean_mask = torch.ones(n, dtype=bool).cuda()
            all_idx = torch.arange(n, device='cuda')

            for i in range(budget):
                projections = torch.matmul(A.T, resid)
                index = torch.argmax(projections[boolean_mask])
                index = all_idx[boolean_mask][index]

                indices.append(index.item())
                boolean_mask[index] = False

                if indices.__len__() == 1:
                    A_i = A[:, index]
                    x_i = projections[index] / torch.dot(A_i, A_i).view(-1)
                    A_i = A[:, index].view(1, -1)
                else:
                    A_i = torch.cat((A_i, A[:, index].view(1, -1)), dim=0)
                    temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + self.lam * torch.eye(A_i.shape[0], device='cuda')
                    x_i, _ = torch.lstsq(torch.matmul(A_i, b).view(-1, 1), temp)
                resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i).view(-1)
            if budget > 1:
                    x_i = nnls(temp.cpu().numpy(), torch.matmul(A_i, b).view(-1).cpu().numpy())[0]
                    x[indices] = x_i
            elif budget == 1:
                x[indices[0]] = 1
        
        return x
    
    def extract_grad_y(self):
        clf = get_classifier(self.arch, self.num_classes, self.pretrain)
        if torch.cuda.is_available():
            clf.cuda()
        cudnn.benchmark = True
        clf.eval()

        grads_all = torch.zeros([0, self.num_classes * (clf.in_features+1)], requires_grad=False).cuda()
        labels_all = torch.zeros([0], requires_grad=False).cuda()

        for data in self.data_loader:
            inputs, labels = data
            inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

            with torch.no_grad():
                z, h = clf(inputs, ret_feat=True)
            
            loss = F.cross_entropy(h.requires_grad_(True), labels).sum()
            bias_parameters_grads = torch.autograd.grad(loss, h)[0]
            weight_parameters_grads = z.view(len(labels), 1, clf.in_features).repeat(1, self.num_classes, 1) * bias_parameters_grads.view(len(labels), self.num_classes, 1).repeat(1, 1, clf.in_features)
        
            
            grads = torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1)
            grads_all = torch.cat((grads_all, grads), dim=0)
            labels_all = torch.cat((labels_all, labels), dim=0)
        
        return grads_all, labels_all
    
    def select_in_category(self, c):
        idxs_c = torch.where(self.labels == c)[0].cpu().numpy()
        num_c = len(idxs_c)
        num_selected_c = round(num_c * self.ratio)

        grads_c = self.grads[idxs_c]
        cur_val_gradients = torch.mean(grads_c, dim=0)
        cur_weights = self.orthogonal_matching_pursuit(grads_c.T, cur_val_gradients, budget=num_selected_c)
        
        return idxs_c[np.nonzero(cur_weights)[0]]
