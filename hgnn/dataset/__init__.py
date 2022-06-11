import importlib
from dgl.data import DGLDataset
from .base_dataset import BaseDataset
from .adapter import AsNodeClassificationDataset

DATASET_REGISTRY = {}

def register_dataset(name):
    """
    New dataset types can be added to cogdl with the :func:`register_dataset`
    function decorator.

    For example::

        @register_dataset('my_dataset')
        class MyDataset():
            (...)

    Args:
        name (str): the name of the dataset
    """

    def register_dataset_cls(cls):
        if name in DATASET_REGISTRY:
            raise ValueError("Cannot register duplicate dataset ({})".format(name))
        if not issubclass(cls, BaseDataset):
            raise ValueError("Dataset ({}: {}) must extend cogdl.data.Dataset".format(name, cls.__name__))
        DATASET_REGISTRY[name] = cls
        return cls

    return register_dataset_cls


def try_import_task_dataset(task):
    if task not in DATASET_REGISTRY:
        if task in SUPPORTED_DATASETS:
            importlib.import_module(SUPPORTED_DATASETS[task])
        else:
            print(f"Failed to import {task} dataset.")
            return False
    return True


def build_dataset(dataset, task, *args, **kwargs):
    if isinstance(dataset, DGLDataset):
        return dataset
    if dataset in CLASS_DATASETS:
        return build_dataset_v2(dataset, task)
    if not try_import_task_dataset(task):
        exit(1)
    _dataset = 'rdf_' + task
    
    return DATASET_REGISTRY[_dataset](dataset, logger=kwargs['logger'])


SUPPORTED_DATASETS = {
    "node_classification": "openhgnn.dataset.NodeClassificationDataset",
}

from .NodeClassificationDataset import NodeClassificationDataset


def build_dataset_v2(dataset, task):
    if dataset in CLASS_DATASETS:
        path = ".".join(CLASS_DATASETS[dataset].split(".")[:-1])
        module = importlib.import_module(path)
        class_name = CLASS_DATASETS[dataset].split(".")[-1]
        dataset_class = getattr(module, class_name)
        d = dataset_class()
        if task == 'node_classification':
            target_ntype = getattr(d, 'category')
            if target_ntype is None:
                target_ntype = getattr(d, 'target_ntype')
            res = AsNodeClassificationDataset(d, target_ntype=target_ntype)
        return res


CLASS_DATASETS = {
    "dblp4GTN": "openhgnn.dataset.DBLP4GTNDataset",
    "acm4GTN": "openhgnn.dataset.ACM4GTNDataset",
    "imdb4GTN": "openhgnn.dataset.IMDB4GTNDataset",
}

__all__ = [
    'BaseDataset',
    'NodeClassificationDataset',
    'AsNodeClassificationDataset'
]

classes = __all__