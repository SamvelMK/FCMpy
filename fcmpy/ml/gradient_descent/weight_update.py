from abc import ABC, abstractmethod


class WeightUpdate(ABC):
    """
        Interface for updating the parameter matrix W.
    """
    @abstractmethod
    def update(**kwargs):
        raise NotImplementedError('update method is not defined.')

