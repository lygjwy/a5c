import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from .base_select import BaseSelect
from models import get_cclf

class Forgetting(BaseSelect):

    def __init__(self, num_classes, data_loader, models):
        super().__init__(num_classes, data_loader)

        self.models = models
        self.forgetting, self.labels = self.extract_forgetting_y()

    def extract_forgetting_y(self):

        # record acc from different epoch clf
        accs_all = []
        for i in range(len(self.models)-1):
            accs_all.append(np.array([]))
        # for i in range(5):
        #     accs_all.append(np.array([]))
        
        # traverse the models
        for i, model in enumerate(self.models[1:]):
            # remove the random init model
            arch, ckpt = model
            cclf = get_cclf(arch, self.num_classes, ckpt)
            if torch.cuda.is_available():
                cclf.cuda()
            cudnn.benchmark = True

            # extract the classifier's prediction
            cclf.eval()

            labels_all = torch.zeros([0], requires_grad=False).cuda()
            for data in self.data_loader:
                inputs, labels = data
                inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

                with torch.no_grad():
                    h = cclf(inputs)
                    _, preds = torch.max(h, dim=1)

                accs_all[i] = np.append(accs_all[i], (preds == labels).cpu().numpy())

                labels_all = torch.cat((labels_all, labels), dim=0)

        # calculate forgetting with accs_all
        forgetting = np.zeros_like(accs_all[0])
        for i in range(len(accs_all)-1):
            forgetting += (accs_all[i] - accs_all[i+1]) > 0.01

        return forgetting, labels_all

    def select_in_category(self, c, ratio):
        
        idxs_c = torch.where(self.labels == c)[0].cpu().numpy()
        num_c = len(idxs_c)

        scores_c = self.forgetting[idxs_c]
        num_selected_c = round(num_c * ratio)
        
        return idxs_c[np.argsort(scores_c)[::-1][:num_selected_c]]