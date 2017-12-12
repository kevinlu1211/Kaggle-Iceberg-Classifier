from .AbstractEvaluationDelegate import AbstractEvaluationDelegate
import torch
from pathlib import Path
import pandas as pd
from glob import glob


class EvaluationDelegate(AbstractEvaluationDelegate):

    def __init__(self, experiment_id, experiment_path):
        self.experiment_id = experiment_id
        self.experiment_path = experiment_path
        self.data_ids = []
        self.model_outputs = []

    def on_setup_model(self, model):
        model_checkpoint_path = f"{self.experiment_path}/{self.experiment_id}/model_checkpoints"
        model_checkpoint_name = self.get_checkpoint_name(model_checkpoint_path)
        model = model.load_state_dict(torch.load(f"{model_checkpoint_path}/{model_checkpoint_name}.pth"))
        return model

    def get_checkpoint_name(self, model_checkpoint_path):
        checkpoints = glob.glob(f"{model_checkpoint_path}/*.pth")
        losses = [float(loss.split(".")[0]) for loss in checkpoints]
        lowest_loss = min(losses)
        return f"{lowest_loss}.pth"

    def on_test_start(self, dataset):
        return dataset['test']

    def save_evaluation_data(self, data, model_output):
        self.data_ids.extend(data['id'])
        self.model_outputs.extend(model_output.view(-1).data.numpy().tolist())

    def write_to_csv(self):
        res = []
        for id, prob in zip(self.data_ids, self.model_outputs):
            res.append({"id": id,
                        "is_iceberg": prob})
        df = pd.DataFrame(res)
        output_save_path = Path(f"{self.experiment_path}/{self.id}/model_outputs")
        output_save_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_save_path, index=False)

    def on_test_end(self):
        self.write_to_csv()
