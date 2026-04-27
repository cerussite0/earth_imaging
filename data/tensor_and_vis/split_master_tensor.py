
import os, argparse
import torch

def split_tensor(input_path, ratio, dimension):
    data = torch.load(input_path, weights_only=True)
    keys = list(data.keys())
    (_, H, W) = data[keys[0]].shape
    if (dimension == 'H'):
        idx = int((H * ratio))
        train = {k: v[:, :idx, :] for (k, v) in data.items()}
        val = {k: v[:, idx:, :] for (k, v) in data.items()}
    elif (dimension == 'W'):
        idx = int((W * ratio))
        train = {k: v[:, :, :idx] for (k, v) in data.items()}
        val = {k: v[:, :, idx:] for (k, v) in data.items()}
    base = input_path.replace('.pt', '')
    train_path = f'{base}_train.pt'
    val_path = f'{base}_val.pt'
    torch.save(train, train_path)
    torch.save(val, val_path)
    t = train[keys[0]]
    v = val[keys[0]]
    print(f'Train: {t.shape[1]}x{t.shape[2]} -> {train_path}')
    print(f'Val:   {v.shape[1]}x{v.shape[2]} -> {val_path}')
if (__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_tensor', type=str, required=True)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--split_dim', type=str, choices=['H', 'W'], default='H')
    args = parser.parse_args()
    split_tensor(args.input_tensor, args.train_ratio, args.split_dim)
