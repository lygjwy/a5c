import torch
from .geo_select import GeoSelect


class kCenterGreedy(GeoSelect):

    def __init__(self, num_classes, data_loader, models, metric='euclidean'):
        super().__init__(num_classes, data_loader, models, metric)

    def k_center_greedy(self, idxs, budget):
        num = len(idxs)
        z = self.zs[idxs]
        status_selected = torch.zeros(num, dtype=bool, requires_grad=False)

        # randomly select one as initial point
        status_selected[torch.randint(0, num, (1,))] = True
        
        # Initialize a budget*sample_num matrix storing distances of pool points from each clustering center.
        dis_matrix = -1 * torch.ones([budget, num]).cuda()
        dis_matrix[0, ~status_selected] = self.metric(z[status_selected], z[~status_selected])
        mins = torch.min(dis_matrix[:1, :], dim=0).values
        
        for i in range(budget-1):
            p = torch.argmax(mins).item()
            status_selected[p] = True

            if i == budget - 2:
                break
            mins[p] = -1 # maybe useless
            dis_matrix[1 + i, ~status_selected] = self.metric(z[[p]], z[~status_selected])
            mins = torch.min(mins, dis_matrix[1 + i])

        return status_selected

    def select_in_category(self, c, ratio):
        idxs_c = torch.where(self.ys == c)[0]
        budget = round(len(idxs_c) * ratio)
        status_selected = self.k_center_greedy(idxs_c, budget)    

        return idxs_c[status_selected].cpu().numpy()