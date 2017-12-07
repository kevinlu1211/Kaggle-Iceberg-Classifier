import argparse
from torch.nn import functional as F
import os
import torch
from time import strftime
import logging
from torch.autograd import Variable
from models.quant_scientist_net import Net
from models.densenet import DenseNet
from utils.dataset import create_train_val_dataloaders
from tqdm import tqdm
use_cuda = torch.cuda.is_available()


def main():
    train_loader, val_loader = create_train_val_dataloaders(args.train_data_fp, batch_size=args.batch_size)
    model = DenseNet(drop_prob=0)
    if use_cuda:
        model = model.cuda()
    train(model, train_loader, val_loader)


def train(model, train_loader, val_loader):
    all_losses = []
    val_losses = []

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-5)  # L2 regularization

    for epoch in range(args.num_epochs):
        logging.info("Epoch {}".format(epoch + 1))
        losses_for_epoch = []
        for i, data in enumerate(tqdm(train_loader), 1):

            (img, label), _ = data
            if use_cuda:
                img, label = Variable(img.cuda(async=True)), Variable(label.cuda(async=True))  # On GPU
            else:
                img, label = Variable(img), Variable(
                    label)  # RuntimeError: expected CPU tensor (got CUDA tensor)

            out = model(img)
            out = F.sigmoid(out)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_for_epoch.append(loss.data[0])

        avg_loss_for_epoch = sum(losses_for_epoch)/len(losses_for_epoch)
        all_losses.append(avg_loss_for_epoch)
        logging.info(f"Finish {epoch+1} epoch, Average Loss: {avg_loss_for_epoch:.3f}")

        model.eval()
        eval_losses_for_epoch = []
        for data in val_loader:
            (img, label), _ = data

            if use_cuda:
                img, label = Variable(img.cuda(async=True), volatile=True), \
                             Variable(label.cuda(async=True), volatile=True)  # On GPU
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)

            out = model(img)
            out = F.sigmoid(out)
            loss = criterion(out, label)
            eval_losses_for_epoch.append(loss.data[0])

        avg_eval_loss_for_epoch = sum(eval_losses_for_epoch)/len(eval_losses_for_epoch)
        logging.info(f"VALIDATION Loss: {avg_eval_loss_for_epoch:.3f}")
        val_losses.append(avg_eval_loss_for_epoch)

        # if len(val_losses) >= 2:
        #     if val_losses[-1] > val_losses[-2]:
        #         break

    os.makedirs(args.save_model_dir, exist_ok=True)
    # torch.save(model.state_dict(), os.path.join(args.save_model_dir, "cnn.pth"))
    torch.save(model.state_dict(), os.path.join(args.save_model_dir, strftime("%H%M-%Y-%m-%d")+'.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", default="densenet")
    parser.add_argument("--train_data_fp", default="data/train.json")
    parser.add_argument("--test_data_fp", default="data/test.json")
    parser.add_argument("--save_model_dir", default="model_checkpoint")
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--num_epochs", default=128)
    parser.add_argument("--learning_rate", default=0.00005)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(level=log_level)
    main()