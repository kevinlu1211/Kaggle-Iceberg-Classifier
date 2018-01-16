import torch
from pathlib import Path
import pandas as pd
from glob import glob


class EvaluationDelegate(object):

    def __init__(self, model_load_path, eval_save_path):
        self.model_load_path = model_load_path
        self.eval_save_path = eval_save_path
        self.data_ids = []
        self.model_outputs = []

    def on_setup_model(self, model):
        model.load_state_dict(torch.load(self.model_load_path))
        return model

    def on_test_start(self, dataset):
        return dataset['test']

    def save_evaluation_data(self, data, model_output):
        self.data_ids.extend(data['id'])
        self.model_outputs.extend(model_output.view(-1).cpu().data.numpy().tolist())

    def create_output(self):
        res = []
        for id, prob in zip(self.data_ids, self.model_outputs):
            res.append({"id": id,
                        "is_iceberg": prob})
        df = pd.DataFrame(res)
        return df

    def write_to_csv(self, df, path):
        eval_save_path = Path(path)
        eval_save_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(f"{eval_save_path}/output.csv", index=False)

    def on_test_end(self):
        df = self.create_output()
        if self.eval_save_path is not None:
            self.write_to_csv(df, self.eval_save_path)
