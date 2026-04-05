import argparse
import os
import torch
from torch.optim import SGD, Adam
import loss
import tools
from dataset import General_Dataset
import tqdm
LOSS_REGISTRY={

}
OPTIMIZER_REGISTRY= {
    "sgd":SGD,
    "adam":Adam,
}

def train(args):
    optim_kwargs = {
        "lr":args.learning_rate,
        "weight_decay":args.weight_decay,
    }
    batch_size = args.batch_size
    epochs = args.epochs
    train_steps = args.train_steps
    accuracy = args.accuracy
    if args.loss not in LOSS_REGISTRY:
        raise ValueError("Unknown loss {}".format(args.loss))
    if args.optimizer == 'sgd':
        optim_kwargs["momentum"] = args.momentum
        optim_kwargs["nesterov"] = True
    elif args.optimizer == 'adam':
        # Adam 不需要 momentum，它用 betas
        # 假设你想把 args.momentum 作为 beta1
        optim_kwargs["betas"] = (args.momentum, 0.999)
    model = None
    criterion=LOSS_REGISTRY[args.loss].cuda()
    optimizer = OPTIMIZER_REGISTRY[args.optimizer](model.parameters(),**optim_kwargs)
    train_dataset=General_Dataset(args.path)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
    test_dataset=General_Dataset(args.test_path)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
    optimizer.zero_grad()
    for epoch in range(epochs):
        total_loss = 0
        train_pbar = tqdm.tqdm(train_dataloader, total=train_steps, desc=f"Epoch [{epoch + 1}/{epochs}]")
        for i,(data,labels) in enumerate(train_pbar):
            if i > train_steps:
                break
            outputs=model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss+=loss.item()
            if i%10==0:
                print(f"Epoch [{epoch + 1}/{epochs}] Step [{i + 1}/{train_steps}] Loss: {loss.item():.4f}")
        avg_loss = total_loss / train_steps
        print(f"Epoch {epoch+1} 结束! 平均 Loss: {avg_loss:.4f}")
        if epochs%10==0:
            with torch.no_grad():
                test_pbar=tqdm.tqdm(test_dataloader, total=train_steps, desc=f"Epoch [{epoch + 1}/{epochs}]")
                for i,(data,labels) in enumerate(test_pbar):
                    outputs=model(data)







































if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_steps", type=int, default=50,help="number of iterations per epoch")
    parser.add_argument("--loss", type=str,choices=[], default="triplet")#NAN
    parser.add_argument("--optimizer", type=str, choices=['sgd', 'adam'], default="adam")
    parser.add_argument("--gpu", default="0", type=str, help="gpu id you use")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--accuracy", type=float, default="float32")
    parser.add_argument("--train_path", type=str, default="./data/train.csv")
    parser.add_argument("--test_path", type=str, default="./data/test.csv")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tools.set_seed(42)
