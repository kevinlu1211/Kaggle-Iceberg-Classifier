import os
import pandas as pd
import argparse
from tqdm import tqdm
import torch
from time import strftime
from torch.autograd import Variable
from utils.dataset import create_dataloader_from_path
from models.quant_scientist_net import Net
use_cuda = False


def main():
    test_dataloader = create_dataloader_from_path(args.test_data_fp)
    model = Net()
    model.load_state_dict(torch.load(args.model_fp))

    # put model in eval model to disable dropout
    model.eval()
    evaluate(model, test_dataloader)


def write_to_csv(ids, probs):
    res = []
    for id, prob in zip(ids, probs):
        res.append({"id": id,
                    "is_iceberg": prob})
    df = pd.DataFrame(res)
    os.makedirs(args.output_fp.split("/")[0], exist_ok=True)
    df.to_csv(args.output_fp, index=False)


def evaluate(model, test_dataloader):
    probs = []
    ids = []
    for data in tqdm(test_dataloader):
        (img, label), id = data
        if use_cuda:
            img, label = Variable(img.cuda(async=True)), Variable(label.cuda(async=True))
        else:
            img, label = Variable(img), Variable(label)

        out = model(img)
        probs.extend(out.view(-1).data.numpy().tolist())
        ids.extend(id)
    write_to_csv(ids, probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_fp", default="data/test.json")
    parser.add_argument("--model_fp", default="model_checkpoint/cnn.pth")
    parser.add_argument("--output_fp", default=f"output/{strftime('%H%M-%Y-%m-%d')}.csv")
    args = parser.parse_args()
    main()
