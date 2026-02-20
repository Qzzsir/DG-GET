from source.dataset import dataset_factory1

# from omegaconf import DictConfig, open_dict
# from .abide import load_abide_data
# from .dataloader import (
#     init_site_holdout_test_discard,
#     init_whole_site_kfold
# )
# from typing import List
# import torch.utils as utils
#
# def dataset_factory(cfg: DictConfig, fold_id: int = 0) -> List[utils.data.DataLoader]:
#     datasets = load_abide_data(cfg)
#     final_timeseires = datasets[0]
#     final_pearson = datasets[1]
#     labels = datasets[2]
#     site = datasets[3]
#
#
#     if cfg.dataset.mode == "kfold":
#         return init_whole_site_kfold(cfg,
#                                      final_timeseires,
#                                      final_pearson,
#                                      labels,
#                                      site,
#                                      fold_id=fold_id)
#     else:
#         # LOSO
#         return init_site_holdout_test_discard(cfg, final_timeseires, final_pearson, labels, site)
#