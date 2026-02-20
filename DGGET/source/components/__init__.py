from .logger import initialize_logger, logger_factory
from .lr_scheduler import LRScheduler, lr_scheduler_factory
from .optimizer import optimizer_factory, optimizers_factory
"""逐行注释
from .logger import initialize_logger, logger_factory

从 logger 模块中导入两个函数：initialize_logger 和 logger_factory。
initialize_logger：通常用于设置日志记录器的初始化配置，比如设置日志级别、输出格式等。
logger_factory：用于创建和配置日志记录器的工厂函数，通常返回一个已配置好的日志记录器对象。
from .lr_scheduler import LRScheduler, lr_scheduler_factory

从 lr_scheduler 模块中导入两个组件：LRScheduler 类和 lr_scheduler_factory 函数。
LRScheduler：通常是一个用于学习率调度的类，可能包含调整学习率的方法，如基于训练进度调整学习率等。
lr_scheduler_factory：用于创建和配置学习率调度器的工厂函数，通常返回一个根据配置创建的学习率调度器对象。
from .optimizer import optimizer_factory, optimizers_factory

从 optimizer 模块中导入两个函数：optimizer_factory 和 optimizers_factory。
optimizer_factory：用于创建单个优化器的工厂函数，可能根据给定的配置返回不同类型的优化器（如 SGD、Adam 等）。
optimizers_factory：用于创建多个优化器的工厂函数，通常用于需要多个优化器的训练过程，比如在多任务学习中，每个任务可能有自己的优化器。
输入输出维度说明
initialize_logger

输入：通常包括日志配置参数。
输出：配置好的日志记录器对象。
logger_factory

输入：包含日志配置的配置对象。
输出：配置好的日志记录器对象。
LRScheduler

输入：通常包括学习率调度的配置参数。
输出：学习率调度器对象。
lr_scheduler_factory

输入：学习率调度配置对象。
输出：根据配置创建的学习率调度器对象。
optimizer_factory

输入：包括优化器类型和配置参数。
输出：创建的优化器对象。
optimizers_factory

输入：模型对象和优化器配置列表。
输出：包含多个优化器的对象（例如，在多任务学习中，可能会有一个优化器对象的列表）。"""