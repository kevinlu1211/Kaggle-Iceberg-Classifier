from src.Experiment import EvaluationDelegate

class DenseNetEvaluationDelegate(EvaluationDelegate):
    def __init__(self, experiment_id, experiment_path):
        super().__init__(experiment_id, experiment_path)
