import hydra
from datetime import datetime
from omegaconf import DictConfig, open_dict
from source.dataset.dataset_factory1 import dataset_factory
from source.models.model_factory1 import model_factory
from source.components import lr_scheduler_factory, optimizers_factory, logger_factory
from source.training.training_factory import training_factory


def model_training_10cv(cfg: DictConfig, fold_id: int = 0):
    with open_dict(cfg):
        cfg.unique_id = f"{datetime.now().strftime('%m-%d-%H-%M-%S')}_fold{fold_id}"
    dataloaders = dataset_factory(cfg, fold_id=fold_id)
    logger = logger_factory(cfg)
    model = model_factory(cfg)
    optimizers = optimizers_factory(model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)
    training = training_factory(cfg, model, optimizers, lr_schedulers, dataloaders, logger)
    metrics = training.train()
    return metrics


# LOSO
def model_training_loso(cfg: DictConfig, run_id: int = 0):
    with open_dict(cfg):
        cfg.unique_id = f"{datetime.now().strftime('%m-%d-%H-%M-%S')}_run{run_id}"
    dataloaders = dataset_factory(cfg)
    logger = logger_factory(cfg)
    model = model_factory(cfg)
    optimizers = optimizers_factory(model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)
    training = training_factory(cfg, model, optimizers, lr_schedulers, dataloaders, logger)
    metrics = training.train()
    return metrics


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    selected_metrics = ["Test Accuracy", "Test AUC", "Test Sensitivity", "Test Specificity"]
    all_metrics = {k: [] for k in selected_metrics}

    mode = "LOSO"
    if mode == "kfold":
        # Whole-site 10-fold
        all_fold_metrics = []
        for fold in range(10):
            print(f"=== Fold {fold + 1}/10 ===")
            metrics = model_training_10cv(cfg, fold_id=fold)
            all_fold_metrics.append(metrics)

        avg_metrics = {}
        for key in all_fold_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_fold_metrics) / 10

        print("\n=== 10-Fold Average Results ===")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.4f}")

        return avg_metrics
    else:
        # LOSO
        if cfg.repeat_time <= 0:
            raise ValueError("配置文件中 repeat_time 必须大于0！")

        for run_id in range(cfg.repeat_time):
            print(f"=== Training run {run_id + 1}/{cfg.repeat_time} ===")
            metrics = model_training_loso(cfg, run_id)
            for k in selected_metrics:
                if k in metrics:
                    all_metrics[k].append(metrics[k])
                else:
                    print(f"警告：当前轮次未找到 {k} 指标，跳过")

        avg_metrics = {}
        for k, v in all_metrics.items():
            if len(v) > 0:
                avg_metrics[k.replace("Test ", "")] = sum(v) / len(v)
            else:
                avg_metrics[k.replace("Test ", "")] = 0.0
                print(f"警告：{k} 无有效数据，平均值设为0")

        print(f"\n=== Average Metrics over {cfg.repeat_time} runs ===")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.4f}")
    return avg_metrics


if __name__ == '__main__':
    main()