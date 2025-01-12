import torch
from .geo_select import GeoSelect


class HerdingSelect(GeoSelect):

    def __init__(self, num_classes, data_loader, models, metric='euclidean'):
        super().__init__(num_classes, data_loader, models, metric)

    def herding(self, idxs, budget):
        num = len(idxs)
        seqs = torch.arange(num)
        z = self.zs[idxs]
        mu = torch.mean(z, dim=0)
        status_selected = torch.zeros(num, dtype=bool, requires_grad=False)

        for i in range(budget):
            dist = self.metric(((i + 1) * mu - torch.sum(z[status_selected], dim=0)).view(1, -1), z[~status_selected]) # [1, 800]
            p = torch.argmax(dist)
            p = seqs[~status_selected][p]
            status_selected[p] = True

        return status_selected
    
    def select_in_category(self, c, ratio):
        idxs_c = torch.where(self.ys == c)[0]
        budget = round(len(idxs_c) * ratio)
        status_selected = self.herding(idxs_c, budget)

        return idxs_c[status_selected].cpu().numpy()
