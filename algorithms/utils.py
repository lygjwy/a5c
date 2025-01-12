import numpy as np

from .random import RandomSelect
from .herding import HerdingSelect
from .kcentergreedy import kCenterGreedy
# from .maha import Maha
from .uncertainty import Uncertainty
from .forgetting import Forgetting
from .memory import Memory
from .grand import GraNd
from .el2n import EL2N
from .contextualdiversity import ContextualDiversity
from .cal import Cal
from .cumu_loss import CumuLoss

def get_selector(principle, num_classes, data_loader, models):
    
    if principle == 'random':
        return RandomSelect(num_classes, data_loader)
    
    if principle in ['herding', 'kcentergreedy', 'contextualdiversity']:
        if principle == 'herding':
            return HerdingSelect(num_classes, data_loader, models)
        elif principle == 'kcentergreedy':
            return kCenterGreedy(num_classes, data_loader, models)
        elif principle == 'contextualdiversity':
            return ContextualDiversity(num_classes, data_loader, models)
        else:
            raise ValueError('NOT SUPPORTED PRINCIPLE')
    # elif principle in ['maha_a', 'maha_t']:
    #     return Maha(num_classes, data_loader, models, principle)
    elif principle in ['leastconfidence', 'entropy', 'margin']:
        return Uncertainty(num_classes, data_loader, models, principle)
    elif principle in ['forgetting', 'memory']:
        if principle in 'fogetting':
            return Forgetting(num_classes, data_loader, models)
        else:
            return Memory(num_classes, data_loader, models)
    elif principle in ['cle', 'clh']:
        return CumuLoss(num_classes, data_loader, models, principle)
    elif principle in ['grand', 'el2n']:
        if principle == 'grand':
            return GraNd(num_classes, data_loader, models)
        elif principle == 'el2n':
            return EL2N(num_classes, data_loader, models)
        else:
            raise ValueError('NOT SUPPORTED PRINCIPLE')
    # elif principle in ['deepfool', 'cal']:
    #     if principle == 'cal':
    #         return Cal(num_classes, data_loader, models)
    #     else:
    #         raise ValueError('NOT SUPPORTED PRINCIPLE')
    else:
        raise ValueError('NOT SUPPORTED PRINCIPLE')
    

def get_submodular_optimizer(greedy, index, budget, already_selected):
    if greedy == 'naive':
        return NaiveGreedy(index, budget, already_selected)
    elif greedy == 'lazy':
        return LazyGreedy(index, budget, already_selected)
    elif greedy == 'stochastic':
        return StochasticGreedy(index, budget, already_selected)
    elif greedy == 'approximate':
        return ApproximateLazyGreedy(index, budget, already_selected)
    else:
        raise ValueError('ILLEGAL GREEDY OPTION')

# def get_submodular_func
def get_submodular_function(func, index, similarity_kernel=None, similarity_matrix=None, already_selected=[]):
    if func == 'facilitylocation':
        return FacilityLocation(index, similarity_kernel, similarity_matrix, already_selected)
    elif func == 'graphcut':
        return GraphCut(index, similarity_kernel, similarity_matrix, already_selected)
    elif func == 'logdeterminant':
        return LogDeterminant(index, similarity_kernel, similarity_matrix, already_selected)
    else:
        raise ValueError('ILLEGAL SUBMODULAR FUNC')


# submodular functions
class SubmodularFunction(object):

    def __init__(self, index, similarity_kernel=None, similarity_matrix=None, already_selected=[]):
        self.index = index
        self.n = len(index)

        self.already_selected = already_selected

        assert similarity_kernel is not None or similarity_matrix is not None
        if similarity_kernel is not None:
            assert callable(similarity_kernel)
            self.similarity_kernel = self._similarity_kernel(similarity_kernel)
        else:
            assert similarity_matrix.shape[0] == self.n and similarity_matrix.shape[1] == self.n
            self.similarity_matrix = similarity_kernel
            self.similarity_kernel = lambda a, b: self.similarity_matrix[np.ix_(a, b)]

    def _similarity_kernel(self, similarity_kernel):
        return similarity_kernel
    

class FacilityLocation(SubmodularFunction):

    def __init__(self, index, similarity_kernel=None, similarity_matrix=None, already_selected=[]):
        super().__init__(index, similarity_kernel, similarity_matrix, already_selected)

        if self.already_selected.__len__() == 0:
            self.cur_max = np.zeros(self.n, dtypes=np.float32)
        else:
            self.cur_max = np.max(self.similarity_kernel(np.arange(self.n), self.already_selected), axis=1)

        self.all_idx = np.ones(self.n, dtype=bool)

    def _similairty_kernel(self, similarity_kernel):
        self.sim_matrix = np.zeros([self.n, self.n], dtype=np.float32)
        self.if_columns_calculated = np.zeros(self.n, dtype=bool)

        def _func(a, b):
            if not np.all(self.if_columns_calculated[b]):
                if b.dtype != bool:
                    temp = ~self.all_idx
                    temp[b] = True
                    b = temp
                not_calculated = b & ~self.if_columns_calculated
                self.sim_matrix[:, not_calculated] = similarity_kernel(self.all_idx, not_calculated)
                self.if_columns_calculated[not_calculated] = True
            return self.sim_matrix[np.ix_(a, b)]
        
        return _func
        
    def calc_gain(self, idx_gain, selected):
        gains = np.maximum(0, self.similarity_kernel(self.all_idx, idx_gain) - self.cur_max.reshape(-1, 1)).sum(axis=0)
        
        return gains
    
    def calc_gain_batch(self, idx_gain, selected, batch):
        batch_idx = ~self.all_idx
        batch_idx[0:batch] = True
        gains = np.maximum(0., self.similarity_kernel(batch_idx, idx_gain) - self.cur_max[batch_idx].reshape(-1, 1)).sum(axis=0)
        for i in range(batch, self.n, batch):
            batch_idx = ~self.all_idx
            batch_idx[i * batch:(i+1) * batch] = True
            gains += np.maximum(0., self.similarity_kernel(batch_idx, idx_gain) - self.cur_max[batch_idx].reshape(-1, 1)).sum(axis=0)
        
        return gains

    def update_state(self, new_selection, total_selected):
        self.cur_max = np.maximum(self.cur_max, np.max(self.similarity_kernel(self.all_idx, new_selection), axis=1))

class GraphCut(SubmodularFunction):
    
    def __init__(self, index, similarity_kernel=None, similarity_matrix=None, already_selected=[], lam: float=1.):
        super().__init__(index, similarity_kernel, similarity_matrix, already_selected)
        self.lam = lam

        if self.similarity_matrix is not None:
            self.sim_matrix_cols_sum = np.sum(self.similarity_matrix, axis=0)
        self.all_idx = np.ones(self.n, dtype=bool)

    def _similairty_kernel(self, similarity_kernel):
        self.sim_matrix = np.zeros([self.n, self.n], dtype=np.float32)
        self.sim_matrix_cols_sum = np.zeros(self.n, dtype=np.float32)
        self.if_columns_calculated = np.zeros(self.n, dtype=bool)

        def _func(a, b):
            if not np.all(self.if_columns_calculated[b]):
                if b.dtype != bool:
                    temp = ~self.all_idx
                    temp[b] = True
                    b = temp
                not_calculated = b & ~self.if_columns_calculated
                self.sim_matrix[:, not_calculated] = similarity_kernel(self.all_idx, not_calculated)
                self.sim_matrix_cols_sum[not_calculated] = np.sum(self.sim_matrix[:, not_calculated], axis=0)
                self.if_columns_calculated[not_calculated] = True
            
            return self.sim_matrix[np.ix_(a, b)]
        
        return _func
    
    def calc_gain(self, idx_gain, selected):

        gain = -2. * np.sum(self.similarity_kernel(selected, idx_gain), axis=0) + self.lam * self.sim_matrix_cols_sum[idx_gain]

        return gain
    

class LogDeterminant(SubmodularFunction):

    def __init__(self, index, similarity_kernel=None, similarity_matrix=None, already_selected=[]):
        super().__init(index, similarity_kernel, similarity_matrix, already_selected)

        self.all_idx = np.ones(self.n, dtype=bool)

    def _similarity_kernel(self, similarity_kernel):
        self.sim_matrix = np.zeros([self.n, self.n], dtype=np.float32)
        self.if_columns_calculated = np.zeros(self.n, dtype=bool)

        def _func(a, b):
            if not np.all(self.if_columns_calculated[b]):
                if b.dtype != bool:
                    temp = ~self.all_idx
                    temp[b] = True
                    b = temp
                not_calculated = b & ~self.if_columns_calculated
                self.sim_matrix[:, not_calculated] = similarity_kernel(self.all_idx, not_calculated)
                self.if_columns_calculated[not_calculated] = True
            
            return self.sim_matrix[np.id_(a, b)]
        
        return _func
    
    def calc_gain(self, idx_gain, selected):
        sim_idx_gain = self.similarity_kernel(selected, idx_gain).T
        sim_selected = self.similarity_kernel(selected, selected)

        return (np.dot(sim_idx_gain, np.linalg.pinv(sim_selected)) * sim_idx_gain).sum(-1)


# Submodular Optimizer

class SubmodularOptimizer(object):

    def __init__(self, index, budget, already_selected=[]):
        self.index = index

        if budget <= 0 or budget > index.__len__():
            raise ValueError('ILLEGAL BUDGET FOR OPTIMIZER')
        
        self.n = len(index)
        self.budget = budget
        self.already_selected = already_selected


class NaiveGreedy(SubmodularOptimizer):

    def __init__(self, index, budget, already_selected=[]):
        super().__init__(index, budget, already_selected)

    def select(self, gain_function, update_state=None):
        assert callable(gain_function)
        if update_state is not None:
            assert callable(update_state)
        selected = np.zeros(self.n, dtype=bool)
        selected[self.already_selected] = True

        greedy_gain = np.zeros(len(self.index))
        for i in range(sum(selected), self.budget):
            greedy_gain[~selected] = gain_function(~selected, selected)
            current_selection = greedy_gain.argmax()
            selected[current_selection] = True
            greedy_gain[current_selection] = -np.inf
            if update_state is not None:
                update_state(np.array([current_selection]), selected)
        
        return self.index[selected]
    

class LazyGreedy(SubmodularOptimizer):

    def __init__(self, index, budget, already_selected=[]):
        super().__init__(index, budget, already_selected)

    def select(self, gain_function, update_state=None):
        assert callable(gain_function)
        if update_state is not None:
            assert callable(update_state)
        selected = np.zeros(self.n, dtype=bool)
        selected[self.already_selected] = True

        greedy_gain = np.zeros(len(self.index))
        greedy_gain[~selected] = gain_function(~selected, selected)
        greedy_gain[selected] = -np.inf

        for i in range(sum(selected), self.budget):
            best_gain = -np.inf
            last_max_element = -1
            while True:
                cur_max_element = greedy_gain.argmax()
                if last_max_element == cur_max_element:
                    selected[cur_max_element] = True
                    greedy_gain[cur_max_element] = -np.inf

                    if update_state is not None:
                        update_state(np.array([cur_max_element]), selected)
                    break
                new_gain = gain_function(np.array([cur_max_element]), selected)[0]
                greedy_gain[cur_max_element] = new_gain
                if new_gain >= best_gain:
                    best_gain = new_gain
                    last_max_element = cur_max_element

        return self.index[selected]
    
class StochasticGreedy(SubmodularOptimizer):

    def __init__(self, index, budget, already_selected=[], epsilon=0.9):
        super().__init__(index, budget, already_selected)
        self.epsilon = epsilon

    def select(self, gain_function, update_state=None):
        assert callable(gain_function)
        if update_state is not None:
            assert callable(update_state)
        selected = np.zeros(self.n, dtype=bool)
        selected[self.already_selected] = True

        sample_size = max(round(-np.log(self.epsilon) * self.n / self.budget), 1)

        greedy_gain = np.zeros(len(self.index))
        all_idx = np.arange(self.n)
        for i in range(sum(selected), self.budget):
            subset = np.random.choice(all_idx[~selected], replace=False, size=min(sample_size, self.n - i))

            if subset.__len__() == 0:
                break
        
            greedy_gain[subset] = gain_function(subset, selected)
            current_selection = greedy_gain[subset].argmax()
            selected[subset[current_selection]] = -np.inf
            if update_state is not None:
                update_state(np.array([subset[current_selection]]), selected)
        
        return self.index[selected]
    
class ApproximateLazyGreedy(SubmodularOptimizer):
    
    def __init__(self, index, budget, already_selected=[], beta=0.9):
        super().__init__(index, budget, already_selected)
        self.beta = beta

    def select(self, gain_function, update_state=None):
        assert callable(gain_function)
        if update_state is not None:
            assert callable(update_state)
        selected = np.zeros(self.n, dtype=bool)
        selected[self.already_selected] = True

        greedy_gain = np.zeros(len(self.index))
        greedy_gain[~selected] = gain_function(~selected, selected)
        greedy_gain[selected] = -np.inf

        for i in range(sum(selected), self.budget):
            while True:
                cur_max_element = greedy_gain.argmax()
                max_gain = greedy_gain[cur_max_element]

                new_gain = gain_function(np.array([cur_max_element]), selected)[0]

                if new_gain >= self.beta * max_gain:
                    selected[cur_max_element] = True
                    greedy_gain[cur_max_element] = -np.inf

                    if update_state is not None:
                        update_state(np.array([cur_max_element]), selected)
                    break
                else:
                    greedy_gain[cur_max_element] = new_gain
        
        return self.index[selected]