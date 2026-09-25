"""
One place that knows which datasets carry several labels, and how to load them.

Every multitask hook in the pipeline (training loaders, probe loaders, the per-task
lambda analysis, the representation dump) used to import mosei_multitask directly and
test `dataset == "mosei_multitask"`. Adding a second multi-label dataset would have meant
duplicating each call site. They now route through here by dataset name instead.

A dataset module must expose: DATASET_NAME, DATASET_CLASS, LABEL_NAMES, USABLE_TASKS,
FEAT_DIM, task_index, ssl_loader(path, batch_size, num_workers, **kw) and
probe_loader(path, split, batch_size, task=None, **kw).
"""
from pareto_ssl.multibench import chsims as _chsims
from pareto_ssl.multibench import mosei_multitask as _mosei_mt

_REGISTRY = {m.DATASET_NAME: m for m in (_mosei_mt, _chsims)}


def is_multitask_dataset(dataset):
    return dataset in _REGISTRY


def module(dataset):
    if dataset not in _REGISTRY:
        raise KeyError(f"'{dataset}' is not a multitask dataset; known: {sorted(_REGISTRY)}")
    return _REGISTRY[dataset]


def usable_tasks(dataset):
    return list(module(dataset).USABLE_TASKS)


def label_names(dataset):
    return list(module(dataset).LABEL_NAMES)


def task_index(dataset, task):
    return module(dataset).task_index(task)


def dataset_class(dataset):
    return module(dataset).DATASET_CLASS


def feat_dim(dataset):
    return dict(module(dataset).FEAT_DIM)


def ssl_loader(dataset, data_path, batch_size, num_workers=4, **kw):
    return module(dataset).ssl_loader(data_path, batch_size, num_workers, **kw)


def probe_loader(dataset, data_path, split, batch_size, task=None, **kw):
    return module(dataset).probe_loader(data_path, split, batch_size, task=task, **kw)
