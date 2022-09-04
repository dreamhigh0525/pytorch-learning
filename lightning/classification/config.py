from dataclasses import dataclass
from enum import Enum

@dataclass
class DataConfig:
    train_filepath: str = '../../data/mnist/train.csv'
    test_filepath: str = '../../data/mnist/test.csv'
    image_dir: str = ' '
    batch_size: int = 100
    train_fraction = 0.8
    num_workers: int = 2


@dataclass
class ModelConfig:
    arch: str = 'mobilenetv3'  # or 'resnet50'
    num_classes: int = 2
    base_lr: int = 1e-4  ## AdamW: 1e-4, SGD: 1e-2
    step_size: int = 25  ## for StepLR
    gamma: float = 0.1


class Phase(Enum):
    TRIAN = 'train'
    VAL = 'val'
    TEST = 'test'