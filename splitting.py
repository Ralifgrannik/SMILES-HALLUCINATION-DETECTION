"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

def split_data(y, df=None, test_size=0.15, val_size=0.15, random_state=42):

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    all_splits = []
    idx = np.arange(len(y))
    
    for train_val_idx, test_idx in skf.split(idx, y):

        v_size = val_size / (1.0 - test_size)
        
        tr_idx, va_idx = train_test_split(
            train_val_idx,
            test_size=v_size,
            random_state=random_state,
            stratify=y[train_val_idx]
        )
        all_splits.append((tr_idx, va_idx, test_idx))
        
    return all_splits