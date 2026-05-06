from .preprocessor import Preprocessor

def __getattr__(name):
    """Lazy imports for torch-dependent modules."""
    if name == 'ScreenShot':
        from .screenshot import ScreenShot
        return ScreenShot
    elif name == 'FoundationModel':
        from .model import FoundationModel
        return FoundationModel
    elif name == 'FoundationModelTrainer':
        from .trainers import FoundationModelTrainer
        return FoundationModelTrainer
    elif name == 'DrugScreenConfig':
        from .utils import DrugScreenConfig
        return DrugScreenConfig
    elif name == 'get_device':
        from .utils import get_device
        return get_device
    elif name in ('TestDataset', 'TrainingDataset', 'MultiEpochTrainingDataset', 'random_int_between'):
        from . import data_loader
        return getattr(data_loader, name)
    raise AttributeError(f"module 'screenshot' has no attribute {name!r}")
