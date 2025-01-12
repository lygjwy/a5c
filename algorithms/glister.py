import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from .base_select import BaseSelect
from .utils import get_submodular_optimizer
# from models import get_classifier


class Glister(BaseSelect):

    def __init__(self, num_classes, data_loader, ratio, models, greedy='lazy', eta=None, val=True, record_val_detail=True):
        super().__init__(num_classes, data_loader, ratio)
        self.arch, self.pretrain = models[-1]
        self.greedy = greedy
        self.eta = eta
        self.val = val
        self.record_val_detail = record_val_detail
        self.train_grads, self.labels = self.extract_grad_y()
        # TODO: poor
        self.batch_size = 64

    def extract_grad_y(self):
        gradients = []
        labels_all = torch.zeros([0], requires_grad=False).cuda()

        clf = get_classifier(self.arch, self.num_classes, self.pretrain)
        if torch.cuda.is_available():
            clf.cuda()
        cudnn.benchmark = True
        clf.eval()

        self.in_features = clf.in_features

        if self.val:
            data_loader = self.data_loader['val']
        else:
            data_loader = self.data_loader['train']

        if self.val and self.record_val_detail:
            self.init_out = []
            self.init_emb = []
            self.init_y = []

        for data in data_loader:
            inputs, labels = data
            inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

            z, h = clf(inputs, ret_feat=True)
            loss = F.cross_entropy(h.requires_grad_(True), labels).sum()
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, h)[0]
                weight_parameters_grads = z.view(len(labels), clf.in_features).repeat(1, self.num_classes, 1) * bias_parameters_grads.view(len(labels), self.num_classes, 1).repeat(1, 1, clf.in_features)
                gradients.append(torch.cat(
                    [bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1).cpu())

                if self.val and self.record_val_detail:
                    self.init_out.append(h.cpu())
                    self.init_emb.append(z.cpu())
                    self.init_y.append(labels)
            
            labels_all = torch.cat((labels_all, labels), dim=0)
        gradients = torch.cat(gradients, dim=0)

        if self.val:
            self.val_grads = torch.mean(gradients, dim=0)
        
        if self.val and self.record_val_detail:
            with torch.no_grad():
                self.init_out = torch.cat(self.init_out, dim=0)
                self.init_emb = torch.cat(self.init_emb, dim=0)
                self.init_y = torch.cat(self.init_y)
        
        return gradients, labels_all
    
    def update_val_gradients(self, selected_for_train):
        sum_selected_train_gradients = torch.mean(self.train_grads[selected_for_train], dim=0)

        new_outputs = self.init_out - self.eta * sum_selected_train_gradients[:self.num_classes].view(1, -1).repeat(self.init_out.shape[0], 1) - self.eta * torch.matmul(self.init_emb, sum_selected_train_gradients[self.num_classes:].view(self.num_classes, -1).T)

        sample_num = new_outputs.shape[0]
        gradients = torch.zeros([sample_num, self.num_classes * (self.in_features + 1)], requires_grad=False)
        i = 0
        while i * self.batch_size < sample_num:
            batch_idx = np.arange(sample_num)[i * self.batch_size:min((i+1)*self.batch_size, sample_num)]
            new_output_batch = new_outputs[batch_idx].clone().detach().requires_grad_(True)
            loss = F.cross_entropy(new_output_batch, self.init_y[batch_idx]).sum()
            batch_num = len(batch_idx)

            bias_parameters_grads = torch.autograd.grad(loss, new_output_batch)[0]
            weight_parameters_grads = self.init_emb[batch_idx].view(batch_num, 1, self.in_features).repeat(1, self.num_classes, 1) * bias_parameters_grads.view(batch_num, self.num_classes, 1).repeat(1, 1, self.in_features)

            gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1).cpu().numpy())
            i += 1
        
        self.val_grads = torch.mean(gradients, dim=0)

    def select_in_category(self, c):
        idxs_c = torch.where(self.labels == c)[0].cpu().numpy()
        num_c = len(idxs_c)
        num_selected_c = round(num_c * self.ratio)

        submod_optimizer = get_submodular_optimizer(self.greedy, idxs_c, num_selected_c)

        return submod_optimizer.select(gain_function=lambda idx_gain, selected: torch.matmul(self.train_grads[idx_gain], self.val_grads.view(-1, 1)).detach().cpu().numpy().flatten(), upadate_state=self.update_val_gradients)
