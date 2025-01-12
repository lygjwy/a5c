'''
Baseline select algorithm: balanced sampling across different categories
'''
import copy
import numpy as np
from abc import abstractmethod


class BaseSelect():

    def __init__(self, num_classes, data_loader):
        
        # if ratio <= 0.0 or ratio > 1.0:
            # raise ValueError('Invalid Pruning Ratio')
        # self.ratio = ratio

        self.num_classes = num_classes
        self.data_loader = data_loader
        # self.idxs = np.array([], dtype=np.int64)

    @abstractmethod
    def select_in_category(self, c, r):
        pass

    def select(self):
        idxs_list = []

        for r in range(1, 10):
            ratio = r / 10
            idxs = np.array([], dtype=np.int64)

            for c in range(self.num_classes):
                idxs = np.append(idxs, self.select_in_category(c, ratio))
        
            # np.save(os.path.join(idxs_dir, str(ratio)), idxs)
            idxs_list.append(copy.deepcopy(idxs))
        
        return idxs_list