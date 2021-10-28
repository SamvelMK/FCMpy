from abc import ABC
from abc import abstractmethod


class GA(ABC):
    @abstractmethod
    def run(**kwargs):
        raise NotImplementedError('run method is not defined.')