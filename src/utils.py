import os

def get_project_root():
    """Get the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_models_dir():
    """Get the models directory, create if it doesn't exist."""
    models_dir = os.path.join(get_project_root(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

def get_model_path(filename):
    """Get full path for a model file."""
    return os.path.join(get_models_dir(), filename)

def get_logs_dir():
    """Get the logs directory, create if it doesn't exist."""
    logs_dir = os.path.join(get_project_root(), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir

def get_plots_dir():
    """Get the plots directory, create if it doesn't exist."""
    plots_dir = os.path.join(get_project_root(), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir