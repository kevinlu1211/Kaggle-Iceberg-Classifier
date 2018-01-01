from src.Experiment import SaverDelegate


class QSNetSaverDelegate(SaverDelegate):
    def __init__(self, experiment_id, study_save_path):
        super().__init__(experiment_id, study_save_path)
