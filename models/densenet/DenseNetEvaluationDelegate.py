from src.Experiment import EvaluationDelegate

class DenseNetEvaluationDelegate(EvaluationDelegate):
    def __init__(self, model_load_path, eval_save_path):
        super().__init__(model_load_path, eval_save_path)
