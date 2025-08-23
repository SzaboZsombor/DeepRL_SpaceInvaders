import os

def get_project_root() -> str:
    """Get the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_models_dir() -> str:
    """Get the models directory, create if it doesn't exist."""
    models_dir = os.path.join(get_project_root(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

def get_model_path(filename: str) -> str:
    """Get full path for a model file."""
    return os.path.join(get_models_dir(), filename)

def get_logs_dir() -> str:
    """Get the logs directory, create if it doesn't exist."""
    logs_dir = os.path.join(get_project_root(), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir

def get_plots_dir() -> str:
    """Get the plots directory, create if it doesn't exist."""
    plots_dir = os.path.join(get_project_root(), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def get_study_storage_path(filename: str) -> str:
    """Get the path for the Optuna study storage."""
    return os.path.join(get_logs_dir(), filename)

def get_tensorboard_logs_dir() -> str:
    """Get an ASCII-safe directory for TensorBoard logs."""
    tb_logs_dir = "C:/temp/tensorboard_logs"
    os.makedirs(tb_logs_dir, exist_ok=True)
    return tb_logs_dir