import numpy as np

def compute_multilabel_top_k_accuracy(targets, outputs, k=30):
    top_k = np.argpartition(outputs, -k)[:, -k:]
    targets_non_zero_sample, targets_non_zero_class = targets.nonzero()
    good = 0
    total = 0
    for i, v in zip(targets_non_zero_sample, targets_non_zero_class):
        if v in top_k[i]:
            good += 1
        total += 1
    
    return good / total

def compute_multilabel_special_top_k_accuracy(targets, outputs):
    top_k = np.argsort(outputs, axis=1)[:, ::-1]
    targets_non_zero_sample, targets_non_zero_class = targets.nonzero()
    targets_non_zero_count = (targets == 1.0).sum(axis=1)
    good = 0
    total = 0
    for i, v in zip(targets_non_zero_sample, targets_non_zero_class):
        if int(v) in top_k[i][0:targets_non_zero_count[i]]:
            good += 1
        total += 1
    
    return good / total