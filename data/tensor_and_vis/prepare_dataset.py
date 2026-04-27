
import os, argparse
import torch
from torch.utils.data import Dataset

class LandsatSlidingDataset(Dataset):

    def __init__(self, tensor_path, window_size=128, stride=128, transform=None):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        data = torch.load(tensor_path, weights_only=True)
        self.X = torch.cat([data['bands'], data['ndvi']], dim=0).float()
        self.y = data['esri'].long()
        (self.channels, self.H, self.W) = self.X.shape
        self.offsets = [(r, c) for r in range(0, ((self.H - window_size) + 1), stride) for c in range(0, ((self.W - window_size) + 1), stride)]

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        (r, c) = self.offsets[idx]
        w = self.window_size
        x_patch = self.X[:, r:(r + w), c:(c + w)]
        y_patch = self.y[:, r:(r + w), c:(c + w)]
        if self.transform:
            (x_patch, y_patch) = self.transform(x_patch, y_patch)
        return (x_patch, y_patch)

class ESRIOnlySlidingDataset(Dataset):

    def __init__(self, tensor_path, window_size=384, stride=384):
        super().__init__()
        data = torch.load(tensor_path, weights_only=True)
        self.y = data['esri'].long()
        (_, self.H, self.W) = self.y.shape
        self.window_size = window_size
        self.offsets = [(r, c) for r in range(0, ((self.H - window_size) + 1), stride) for c in range(0, ((self.W - window_size) + 1), stride)]

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        (r, c) = self.offsets[idx]
        w = self.window_size
        return self.y[:, r:(r + w), c:(c + w)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_tensor', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'torch_dataset', 'dataset_tensors.pt')))
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--save_path', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'torch_dataset', 'torch_sliding_dataset.pt')))
    parser.add_argument('--esri_only', action='store_true')
    args = parser.parse_args()
    if args.esri_only:
        dataset = ESRIOnlySlidingDataset(args.input_tensor, args.window_size, args.stride)
        if (len(dataset) == 0):
            print('0 patches, exiting.')
            return
        Y = torch.stack([dataset[i] for i in range(len(dataset))], dim=0)
        os.makedirs((os.path.dirname(args.save_path) or '.'), exist_ok=True)
        torch.save(torch.utils.data.TensorDataset(Y), args.save_path)
    else:
        dataset = LandsatSlidingDataset(args.input_tensor, args.window_size, args.stride)
        if (len(dataset) == 0):
            print('0 patches, exiting.')
            return
        (x_list, y_list) = zip(*[dataset[i] for i in range(len(dataset))])
        X = torch.stack(x_list, dim=0)
        Y = torch.stack(y_list, dim=0)
        os.makedirs((os.path.dirname(args.save_path) or '.'), exist_ok=True)
        torch.save(torch.utils.data.TensorDataset(X, Y), args.save_path)
    size_mb = (os.path.getsize(args.save_path) / (1024 * 1024))
    print(f'Saved {len(dataset)} patches ({size_mb:.1f} MB) to {args.save_path}')
if (__name__ == '__main__'):
    main()
