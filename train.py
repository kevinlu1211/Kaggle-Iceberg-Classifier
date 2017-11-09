import argparse
import torch
from torch.autograd import Variable
from models.quant_scientist_net import Net
from utils.dataset import create_dataloader


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_fp", default='data/train.json')
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("--num_epochs", default=25)
    parser.add_argument("--learning_rate", default=0.0005)
    args = parser.parse_args()
    return args



def main():
    args = parse_arguments()
    use_cuda = torch.cuda.is_available()
    train_loader, val_loader = create_dataloader(args.train_data_fp, use_cuda, batch_size=args.batch_size)
    model = Net()
    if use_cuda:
        model = model.cuda()
    train(model, train_loader, val_loader, args.num_epochs, args.batch_size, args.learning_rate, use_cuda)

def train(model, train_loader, val_loader, num_epochs, batch_size, learning_rate, use_cuda):
    all_losses = []
    val_losses = []

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-5)  # L2 regularization

    for epoch in range(num_epochs):
        print('Epoch {}'.format(epoch + 1))
        print('*' * 5 + ':')
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader, 1):

            img, label = data
            if use_cuda:
                img, label = Variable(img.cuda(async=True)), Variable(label.cuda(async=True))  # On GPU
            else:
                img, label = Variable(img), Variable(
                    label)  # RuntimeError: expected CPU tensor (got CUDA tensor)

            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.data[0] * label.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                all_losses.append(running_loss / (batch_size * i))
                print('[{}/{}] Loss: {:.6f}'.format(
                    epoch + 1, num_epochs, running_loss / (batch_size * i),
                    running_acc / (batch_size * i)))

        print('Finish {} epoch, Loss: {:.6f}'.format(epoch + 1, running_loss / (len(train_loader))))

        model.eval()
        eval_loss = 0
        eval_acc = 0
        for data in val_loader:
            img, label = data

            if use_cuda:
                img, label = Variable(img.cuda(async=True), volatile=True), \
                             Variable(label.cuda(async=True), volatile=True)  # On GPU
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)

            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.data[0] * label.size(0)

        print('VALIDATION Loss: {:.6f}'.format(eval_loss / (len(val_loader))))
        val_losses.append(eval_loss / (len(val_loader)))
        print()

    torch.save(model.state_dict(), './cnn.pth')


if __name__ == "__main__":
    main()