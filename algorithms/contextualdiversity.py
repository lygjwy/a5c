import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from .base_select import BaseSelect
from models import get_cclf


class ContextualDiversity(BaseSelect):

    def __init__(self, num_classes, data_loader, models):
        super().__init__(num_classes, data_loader)
        self.arch, self.ckpt = models[-1]
        self.metric = self._metric
        self.h_all, self.y_all = self.extract_h_y()

    def _metric(self, a_output, b_output):
        with torch.no_grad():
            # Overload self.metric function for kCenterGreedy Algorithm
            aa = a_output.view(a_output.shape[0], 1, a_output.shape[1]).repeat(1, b_output.shape[0], 1)
            bb = b_output.view(1, b_output.shape[0], b_output.shape[1]).repeat(a_output.shape[0], 1, 1)
            return torch.sum(0.5 * aa * torch.log(aa / bb) + 0.5 * bb * torch.log(bb / aa), dim=2)
    
    def extract_h_y(self):
        # h: y_hat
        h_all = torch.zeros([0, self.num_classes], requires_grad=False).cuda()
        y_all = torch.zeros([0], requires_grad=False).cuda()

        cclf = get_cclf(self.arch, self.num_classes, self.ckpt)
        if torch.cuda.is_available():
            cclf.cuda()
        cudnn.benchmark = True
        cclf.eval()

        for data in self.data_loader:
            inputs, labels = data
            inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

            with torch.no_grad():
                logits = cclf(inputs)
                h = F.softmax(logits, dim=1)
            
            h_all = torch.cat((h_all, h), dim=0)
            y_all = torch.cat((y_all, labels), dim=0)

        return h_all, y_all
    
    def k_center_greedy(self, idxs, budget):
        num = len(idxs)
        h = self.h_all[idxs]
        status_selected = torch.zeros(num, dtype=bool, requires_grad=False)

        status_selected[torch.randint(0, num, (1,))] = True
        
        dis_matrix = -1 * torch.ones([budget, num]).cuda()
        dis_matrix[0, ~status_selected] = self.metric(h[status_selected], h[~status_selected])
        mins = torch.min(dis_matrix[:1, :], dim=0).values

        for i in range(budget-1):
            p = torch.argmax(mins).item()
            status_selected[p] = True

            if i == budget - 2:
                break
            mins[p] = -1 # maybe useless
            dis_matrix[1 + i, ~status_selected] = self.metric(h[[p]], h[~status_selected])
            mins = torch.min(mins, dis_matrix[1 + i])

        return status_selected
    
    def select_in_category(self, c, ratio):
        idxs_c = torch.where(self.y_all == c)[0]
        budget = round(len(idxs_c) * ratio)
        status_selected = self.k_center_greedy(idxs_c, budget)
    
        return idxs_c[status_selected].cpu().numpy()