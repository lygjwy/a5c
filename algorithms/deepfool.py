
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from .base_select import BaseSelect
# from models import get_classifier


class DeepFool(BaseSelect):

    def __init__(self, num_classes, data_loader, models, max_iter=50):
        super().__init__(num_classes, data_loader)

        # choose the last ckpts
        self.arch, self.pretrain = models[-1]
        self.max_iter = max_iter
        self.score, self.labels = self.extract_r_y()

    def extract_r_y(self):
        r = np.zeros([0], dtype=np.float32)
        labels_all = torch.zeros([0], requires_grad=False).cuda()

        clf = get_classifier(self.arch, self.num_classes, self.pretrain)
        if torch.cuda.is_available():
            clf.cuda()
        cudnn.benchmark = True
        clf.eval()

        for p in clf.parameters():
            p.requires_grad_(False)
        
        for data in self.data_loader:
            inputs, labels = data
            inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

            sample_size = inputs.shape[0]
            boolean_mask = np.ones(sample_size, dtype=bool)
            all_idx = np.arange(sample_size)

            r_tol = np.zeros([sample_size, inputs.shape[1] * inputs.shape[2] * inputs.shape[3]])

            cur_inputs = inputs.requires_grad_(True)
            original_shape = inputs.shape[1:]

            clf.no_grad = True
            first_preds = clf(cur_inputs).argmax(dim=1)
            clf.no_grad = False

            for i in range(self.max_iter):
                f_all = clf(cur_inputs)

                w_k = []
                for c in range(self.num_classes):
                    w_k.append(torch.autograd.grad(f_all[:, c].sum(), cur_inputs, retain_graph=True))[0].flatten(1)
                w_k = torch.stack(w_k, dim=0)
                w_k = w_k - w_k[first_preds, boolean_mask[boolean_mask]].unsqueeze(0)
                w_k_norm = w_k.norm(dim=2)

                w_k_norm[first_preds, boolean_mask[boolean_mask]] = 1

                l_all = (f_all - f_all[boolean_mask[boolean_mask], first_preds].unsqueeze(1)).detach().abs() / w_k_norm.T
                l_all[boolean_mask[boolean_mask], first_preds] = np.inf

                l_hat = l_all.argmin(dim=1)
                r_i = l_all[boolean_mask[boolean_mask], l_hat].unsqueeze(1) / w_k_norm[l_hat, boolean_mask[boolean_mask]].T.unsuqeeze(1) * w_k[l_hat, boolean_mask[boolean_mask]] 

                r_tol[boolean_mask] += r_i.cpu().numpy()

                cur_inputs += r_i.reshape([r_i.shape[0]] + list(original_shape))

                clf.no_grad = True
                preds = clf(cur_inputs).argmax(dim=1)
                clf.no_grad = False

                index_unfinished = (preds == first_preds)
                if torch.all(~index_unfinished):
                    break

                cur_inputs = cur_inputs[index_unfinished]
                first_preds = first_preds[index_unfinished]
                boolean_mask[all_idx[boolean_mask][~index_unfinished.cpu().numpy()]] = False

            r = np.concatenate((r,  (r_tol*r_tol).sum(axis=1)), axis=0)
            labels_all = torch.cat((labels_all, labels), dim=0)
            
        return r, labels_all


    def select_in_category(self, c, ratio):
        
        idxs_c = torch.where(self.labels == c)[0].cpu().numpy()
        num_c = len(idxs_c)

        scores_c = self.scores[idxs_c]
        num_selected_c = round(num_c * ratio)
        
        return idxs_c[np.argsort(scores_c)[:num_selected_c]]
