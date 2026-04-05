import argparse
from tool import set































if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--train_steps", type=int, default=50,help="number of iterations per epoch")

    parser.add_argument("--loss", type=str,choices=['infonce','arcface','softmax','triplet'], default="triplet")

    parser.add_argument("--backbone", type=str,choices=['resnet50','resnet101'], default="resnet101")
    parser.add_argument("--optimizer", type=str, choices=['sgd', 'adam'], default="adam")
    parser.add_argument("--gpu", default="0", type=str, help="gpu id you use")
    set_seed()