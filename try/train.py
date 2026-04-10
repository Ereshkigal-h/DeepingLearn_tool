import argparse
import os
import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
import model
import tool.tools as tools
from new_dataset import NLPDataset,load_cornell_dialogue
import tqdm
from tool.evaluator import evaluator

LOSS_REGISTRY = {
    "mse": nn.MSELoss(),
    "bce": nn.BCELoss(),
    "ce": nn.CrossEntropyLoss(),
    "triplet": nn.TripletMarginLoss(),
}
OPTIMIZER_REGISTRY = {"sgd": SGD, "adam": Adam}
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在加载分词器: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    samples=load_cornell_dialogue(args.train_path,args.test_path)
    full_dataset = NLPDataset(samples)
    total_len = len(full_dataset)
    #按照 5:1 划分 (5/6 为训练集，1/6 为测试集)
    train_len = int(total_len * (5 / 6))
    test_len = total_len - train_len
    train_dataset, test_dataset = random_split(full_dataset, [train_len, test_len])
    print(f"数据集划分完成 总量: {total_len} | 训练集: {train_len} | 测试集: {test_len}")

    def collate_fn(batch):
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        src_dict = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        src_ids = src_dict["input_ids"]
        src_mask = src_dict["attention_mask"]
        labels_dict = tokenizer(labels, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        labels_ids = labels_dict["input_ids"]
        labels_mask = labels_dict["attention_mask"]
        tar_input = labels_ids[:, :-1]
        tar_label = labels_ids[:, 1:]
        tar_mask = labels_mask[:, :-1]
        return {
            "src": src_ids,
            "src_mask": src_mask,
            "tar_input": tar_input,
            "tar_mask": tar_mask,
            "tar_label": tar_label
        }
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    evaluators = evaluator(args.evaluator)
    if args.loss not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss {args.loss}")
    optim_kwargs = {"lr": args.learning_rate, "weight_decay": args.weight_decay}
    if args.optimizer == 'sgd':
        optim_kwargs.update({"momentum": args.momentum, "nesterov": True})
    elif args.optimizer == 'adam':
        optim_kwargs.update({"betas": (args.momentum, 0.999)})
    my_model = model.transformer(vocab_size=tokenizer.vocab_size).to(device)
    criterion = LOSS_REGISTRY[args.loss].to(device)
    optimizer = OPTIMIZER_REGISTRY[args.optimizer](my_model.parameters(),ignore_index=tokenizer.pad_token_id,**optim_kwargs)
    for epoch in range(args.epochs):
        my_model.train()
        total_loss = 0
        train_pbar = tqdm.tqdm(train_dataloader, total=args.train_steps, desc=f"Epoch [{epoch + 1}/{args.epochs}]")
        for i, batch_data in enumerate(train_pbar):
            if i >= args.train_steps:
                break
            # inputs 现在是一个字典，包含 'input_ids' 和 'attention_mask'
            # 我们把它移动到 GPU/CPU
            inputs = {k: v.to(device) for k, v in batch_data.items()}
            optimizer.zero_grad()
            outputs = my_model(
                src=batch_data['src'],
                tar=batch_data['tar_input'],
                src_mask=batch_data['src_mask'],
                tar_mask=batch_data['tar_mask']
            )
            loss = criterion(outputs.reshape(-1, tokenizer.vocab_size), batch_data['tar_label'].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 10 == 0:
                train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        avg_loss = total_loss / args.train_steps
        print(f"Epoch {epoch + 1} 结束! 平均 Loss: {avg_loss:.4f}")
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                my_model.eval()
                result_dict = evaluators.evaluate(my_model, test_dataloader, tokenizer=tokenizer, device=device)
                print(f"评估结果: {result_dict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_steps", type=int, default=50)
    parser.add_argument("--loss", type=str, default="ce")
    parser.add_argument("--optimizer", type=str, choices=['sgd', 'adam'], default="adam")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--accuracy", type=str, default="float32")


    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased", help="HuggingFace分词器名称")
    parser.add_argument("--max_length", type=int, default=128, help="文本截断的最大长度")
    parser.add_argument("--data_path", type=str, default="./data/full_data.csv", help="总数据集的路径(用来按5:1划分)")

    parser.add_argument("--evaluator", type=str, nargs='+', default=["MULTI_ACC"])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tools.set_seed(42)
    train(args)