from omegaconf import DictConfig

from source.models.DGGET.DGGET import BrainNetworkTransformer
def model_factory(config: DictConfig):
    """
    根据配置文件创建并返回一个模型实例。

    参数:
    config (DictConfig): 配置文件的 DictConfig 实例，包含模型的配置信息

    返回:
    模型实例：根据配置创建的模型实例
    """

    # 如果配置中的模型名称是 "LogisticRegression" 或 "SVC"，则返回 None
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None

    # 使用 eval 函数根据配置中的模型名称创建模型实例，并返回
    # config.model.name 应该是一个类名的字符串，eval 将其转换为对应的类，并用 config 参数初始化该类
    # 这里假设 config.model.name 是合法的类名且可以用来实例化模型
    return eval(config.model.name)(config)
