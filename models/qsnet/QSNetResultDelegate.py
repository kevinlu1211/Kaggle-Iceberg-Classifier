from src.Experiment import ResultDelegate


class QSNetResultDelegate(ResultDelegate):
    def __init__(self, experiment_id, study_save_path):
        super().__init__(experiment_id, study_save_path)
