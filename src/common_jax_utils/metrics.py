import numpy as np

from common_dl_utils.metrics import Metric, MetricFrequency


class LossStandardDeviation(Metric):
    """ 
    Tracks the standard deviation of the loss in a window of training steps.
    """
    required_kwargs = set({'loss'})
    def __init__(self, window_size:int, frequency:str):
        """ 
        :parameter window_size: The size of the window over which the standard deviation of the loss is computed.
        :parameter frequency: frequency for the MetricCollector. Should be one of:
            'every_batch'
            'every_n_batches'
            'every_epoch'
            'every_n_epochs'
        """
        self.window_size = window_size
        self._losses = []
        self.frequency = MetricFrequency(frequency)

    def compute(self, **kwargs):
        loss = kwargs['loss']
        loss = float(loss)
        self._losses.append(loss)
        window = self._losses[-self.window_size:]
        if len(window) == self.window_size:
            return {f'loss_std_over_{self.window_size}_steps':np.std(window)}
        return {f'loss_std_over_{self.window_size}_steps': None}
