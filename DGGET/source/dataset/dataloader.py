import numpy as np
import torch
import torch.utils.data as utils
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from omegaconf import open_dict

site_names = ['CALTECH', 'CMU', 'KKI', 'LEUVEN_1', 'LEUVEN_2', 'MAX_MUN', 'NYU', 'OLIN', 'PITT', 'SBL', 'SDSU', 'STANFORD', 'TRINITY', 'UCLA_1', 'UCLA_2', 'UM_1', 'UM_2', 'USM', 'YALE']
site_to_index = {name: idx for idx, name in enumerate(site_names)}


def init_site_holdout_test_discard(cfg, final_timeseires, final_pearson, labels, site, target_sites=("USM",), site_names_global=None):
    """
    Use all samples from the target_sites as the test set.
    The remaining sites are split into train/val with an 80%:20% ratio.
    - target_sites: tuple or list, the sites reserved entirely for testing (default ("NYU",)).
    - site_names_global: optional; if provided, used for one-hot encoding and validation.
      Otherwise, the function automatically builds the site list from the input data.
    Returns: [train_loader, val_loader, test_loader]
    """
    if site_names_global is None:
        unique_sites = list(dict.fromkeys(site))
        site_names_local = unique_sites
    else:
        site_names_local = list(site_names_global)

    site_to_index_local = {name: idx for idx, name in enumerate(site_names_local)}

    if site_names_global is not None:
        unknown = sorted(set(site) - set(site_names_local))
        if len(unknown) > 0:
            raise ValueError(f"Found sites that are not declared in site_names_global:{unknown}.")

    try:
        site_indices = torch.tensor([site_to_index_local[s] for s in site], dtype=torch.int64)
    except KeyError as e:
        raise KeyError(f"{e} Not Found! Please check site_names or site_names_global=None")

    target_sites = tuple(target_sites) if not isinstance(target_sites, (list, tuple)) else tuple(target_sites)
    target_indices_list = []
    for s in target_sites:
        if s not in site_to_index_local:
            raise ValueError(f"target_sites {s} not in site_names.")
        idxs = (site_indices == site_to_index_local[s]).nonzero(as_tuple=True)[0]
        if idxs.numel() > 0:
            target_indices_list.append(idxs)
    if len(target_indices_list) == 0:
        raise ValueError(f"target_sites={target_sites} Not Found")

    test_indices = torch.cat(target_indices_list)

    all_indices = torch.arange(len(site_indices), dtype=torch.long)
    mask = torch.ones(len(all_indices), dtype=torch.bool)
    mask[test_indices] = False
    remaining_indices = all_indices[mask]

    if remaining_indices.numel() == 0:
        raise ValueError("No remaining samples found for training/validation.")

    pos_discard_ratio = 0.375
    neg_discard_ratio = 0.375

    if not (0 <= pos_discard_ratio < 1):
        raise ValueError("pos_discard_ratio must in [0,1)")
    if not (0 <= neg_discard_ratio < 1):
        raise ValueError("neg_discard_ratio must in [0,1)")

    keep_mask = torch.ones(len(remaining_indices), dtype=torch.bool)
    unique_site_idxs = torch.unique(site_indices[remaining_indices])

    for s_idx in unique_site_idxs.tolist():
        rel_pos = (site_indices[remaining_indices] == s_idx).nonzero(as_tuple=True)[0]

        site_labels = labels[remaining_indices][rel_pos]
        pos_idx = rel_pos[(site_labels == 1)]
        neg_idx = rel_pos[(site_labels == 0)]
        pos_count = len(pos_idx)
        neg_count = len(neg_idx)

        discard_pos_num = int(pos_count * pos_discard_ratio)
        discard_neg_num = int(neg_count * neg_discard_ratio)
        discard_pos_rel = pos_idx[:discard_pos_num]
        discard_neg_rel = neg_idx[:discard_neg_num]
        discard_rel = torch.cat([discard_pos_rel, discard_neg_rel], dim=0)

        keep_mask[discard_rel] = False

    remaining_indices = remaining_indices[keep_mask]
    remaining_labels = labels[remaining_indices].cpu().numpy()
    s = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_rel_idx, val_rel_idx = next(s.split(np.zeros(len(remaining_indices)), remaining_labels))
    train_indices = remaining_indices[train_rel_idx]
    val_indices = remaining_indices[val_rel_idx]
    num_classes = int(labels.max().item()) + 1
    num_sites = len(site_names_local)

    def make_tensors(indices):
        return (
            final_timeseires[indices],
            final_pearson[indices],
            F.one_hot(labels[indices].to(torch.int64), num_classes=num_classes),
            F.one_hot(site_indices[indices], num_classes=num_sites)
        )

    X_train, P_train, Y_train, S_train = make_tensors(train_indices)
    X_val, P_val, Y_val, S_val = make_tensors(val_indices)
    X_test, P_test, Y_test, S_test = make_tensors(test_indices)

    train_dataset = utils.TensorDataset(X_train, P_train, Y_train, S_train)
    val_dataset = utils.TensorDataset(X_val, P_val, Y_val, S_val)
    test_dataset = utils.TensorDataset(X_test, P_test, Y_test, S_test)

    train_loader = utils.DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)
    val_loader = utils.DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=False)
    test_loader = utils.DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=False)

    with open_dict(cfg):
        cfg.steps_per_epoch = (len(X_train) - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    return [train_loader, val_loader, test_loader]
    


def init_whole_site_kfold(cfg,
                          final_timeseires,
                          final_pearson,
                          labels,
                          site,
                          fold_id=0,
                          n_splits=10):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    labels_np = labels.cpu().numpy()
    all_indices = np.arange(len(labels_np))

    splits = list(skf.split(all_indices, labels_np))
    train_idx, test_idx = splits[fold_id]

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    # 从 train 再划 10% 做 validation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_rel_idx, val_rel_idx = next(sss.split(train_idx, labels_np[train_idx]))

    train_indices = torch.tensor(train_idx[train_rel_idx], dtype=torch.long)
    val_indices = torch.tensor(train_idx[val_rel_idx], dtype=torch.long)
    test_indices = torch.tensor(test_idx, dtype=torch.long)

    num_classes = int(labels.max().item()) + 1

    # Site one-hot encoding
    unique_sites = list(dict.fromkeys(site))
    site_to_index_local = {name: idx for idx, name in enumerate(unique_sites)}
    site_indices = torch.tensor([site_to_index_local[s] for s in site], dtype=torch.int64)
    num_sites = len(unique_sites)

    def make_tensors(indices):
        return (
            final_timeseires[indices],
            final_pearson[indices],
            F.one_hot(labels[indices].to(torch.int64), num_classes=num_classes),
            F.one_hot(site_indices[indices], num_classes=num_sites)
        )

    X_train, P_train, Y_train, S_train = make_tensors(train_indices)
    X_val, P_val, Y_val, S_val = make_tensors(val_indices)
    X_test, P_test, Y_test, S_test = make_tensors(test_indices)

    train_dataset = utils.TensorDataset(X_train, P_train, Y_train, S_train)
    val_dataset = utils.TensorDataset(X_val, P_val, Y_val, S_val)
    test_dataset = utils.TensorDataset(X_test, P_test, Y_test, S_test)

    train_loader = utils.DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True,
                                    drop_last=cfg.dataset.drop_last)
    val_loader = utils.DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)
    test_loader = utils.DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)

    with open_dict(cfg):
        cfg.steps_per_epoch = (len(X_train) - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    print(f"Fold {fold_id+1}/{n_splits}: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    return [train_loader, val_loader, test_loader]


def init_site_holdout_test(cfg, final_timeseires, final_pearson, labels, site, target_sites=("USM",), site_names_global=None):
    """
    Use all samples from target_sites as the test set.
    Split the remaining sites into train/val with an 80:20 ratio.
    - target_sites: tuple or list of sites reserved for testing (default ("USM",))
    - site_names_global: optional; used for one-hot encoding and validation. If None, the site list is auto-generated.
    Returns: [train_loader, val_loader, test_loader]
    """

    if site_names_global is None:
        unique_sites = list(dict.fromkeys(site))
        site_names_local = unique_sites
    else:
        site_names_local = list(site_names_global)

    site_to_index_local = {name: idx for idx, name in enumerate(site_names_local)}

    if site_names_global is not None:
        unknown = sorted(set(site) - set(site_names_local))
        if len(unknown) > 0:
            raise ValueError(f"Found sites not declared in site_names_global: {unknown}. Please update the list or set site_names_global=None to auto-generate.")

    try:
        site_indices = torch.tensor([site_to_index_local[s] for s in site], dtype=torch.int64)
    except KeyError as e:
        raise KeyError(f"Failed to map site names to indices. Site not found: {e}. Check `site_names` or set site_names_global=None.")

    target_sites = tuple(target_sites) if not isinstance(target_sites, (list, tuple)) else tuple(target_sites)
    target_indices_list = []
    for s in target_sites:
        if s not in site_to_index_local:
            raise ValueError(f"Target site {s} is not in the known site list.")
        idxs = (site_indices == site_to_index_local[s]).nonzero(as_tuple=True)[0]
        if idxs.numel() > 0:
            target_indices_list.append(idxs)
    if len(target_indices_list) == 0:
        raise ValueError(f"No samples found in the data for target_sites={target_sites}.")

    test_indices = torch.cat(target_indices_list)  # Global indices for the test set

    all_indices = torch.arange(len(site_indices), dtype=torch.long)
    mask = torch.ones(len(all_indices), dtype=torch.bool)
    mask[test_indices] = False
    remaining_indices = all_indices[mask]

    if remaining_indices.numel() == 0:
        raise ValueError("No remaining sites available for train/validation split (all data is assigned to the test set).")

    remaining_labels = labels[remaining_indices].cpu().numpy()
    s = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_rel_idx, val_rel_idx = next(s.split(np.zeros(len(remaining_indices)), remaining_labels))
    train_indices = remaining_indices[train_rel_idx]
    val_indices = remaining_indices[val_rel_idx]

    num_classes = int(labels.max().item()) + 1
    num_sites = len(site_names_local)

    def make_tensors(indices):
        return (
            final_timeseires[indices],
            final_pearson[indices],
            F.one_hot(labels[indices].to(torch.int64), num_classes=num_classes),
            F.one_hot(site_indices[indices], num_classes=num_sites)
        )

    X_train, P_train, Y_train, S_train = make_tensors(train_indices)
    X_val, P_val, Y_val, S_val = make_tensors(val_indices)
    X_test, P_test, Y_test, S_test = make_tensors(test_indices)

    train_dataset = utils.TensorDataset(X_train, P_train, Y_train, S_train)
    val_dataset = utils.TensorDataset(X_val, P_val, Y_val, S_val)
    test_dataset = utils.TensorDataset(X_test, P_test, Y_test, S_test)

    train_loader = utils.DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)
    val_loader = utils.DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=False)
    test_loader = utils.DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=False)

    with open_dict(cfg):
        cfg.steps_per_epoch = (len(X_train) - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    print(f"Holdout test sites={target_sites} -> test={len(test_indices)}, train={len(train_indices)}, val={len(val_indices)} (train:val ≈ 8:2)")

    return [train_loader, val_loader, test_loader]
