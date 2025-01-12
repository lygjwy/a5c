import numpy as np

import torch
from torch.autograd import Variable

from .base_select import BaseSelect


class RandomSelect(BaseSelect):

    def __init__(self, num_classes, data_loader):
        super().__init__(num_classes=num_classes, data_loader=data_loader)
        self.labels = self.extract_y()

    def extract_y(self):
        labels_all = torch.zeros([0], requires_grad=False).cuda()

        for data in self.data_loader:

            _, labels = data
            labels = Variable(labels.long().cuda())
            labels_all = torch.cat((labels_all, labels), dim=0)
        
        return labels_all

    def select_in_category(self, c, ratio):
        # idxs_c = np.where(self.labels == c)
        idxs_c = torch.where(self.labels == c)[0] # return tensor
        num_selected_c = round(len(idxs_c) * ratio)
        
        return np.random.choice(idxs_c.cpu().numpy(), num_selected_c, replace=False)