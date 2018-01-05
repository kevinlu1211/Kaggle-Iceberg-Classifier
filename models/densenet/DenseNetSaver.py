from src.Experiment import SaverDelegate


class DenseNetSaver(SaverDelegate):
    def __init__(self, experiment_id, study_save_path, generation):
        super().__init__(experiment_id, study_save_path, generation)
