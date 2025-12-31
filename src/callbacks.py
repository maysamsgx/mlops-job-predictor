from xgboost.callback import TrainingCallback
import os

class ModelCheckpoint(TrainingCallback):
    """
    Saves XGBoost model at specified intervals.
    """
    def __init__(self, interval=10, model_name='xgb_model', checkpoint_dir='checkpoints'):
        self.interval = interval
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.interval == 0:
            path = os.path.join(self.checkpoint_dir, f"{self.model_name}_{epoch}.json")
            model.save_model(path)
            print(f"[Checkpoint] Saved {self.model_name} state at epoch {epoch} to {path}")
        return False
