import torch
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F

import numpy as np
import cv2

from pathlib import Path
import pprint

from model import AutoEncoder, AE_CNN


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batchsize", type=int, default=50, help="batchsize")
    parser.add_argument("-e", "--epoch", type=int, default=100, help="iteration")
    parser.add_argument("-g", "--gpu", type=int, default=-1, help="GPU ID")
    parser.add_argument("--graph", default=None, help="computational graph")
    parser.add_argument("--cnn", action="store_true", help="CNN")
    parser.add_argument("--lam", type=float, default=0.0, help="weight decay")
    #parser.add_argument("-m", "--model", default="autoencoder.model", help="model file name")
    parser.add_argument("-r", "--result", default="result", help="result directory")
    args = parser.parse_args()

    pprint.pprint(vars(args))
    main(args)


def main(args):
    # GPUを使うための設定
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu:d}")
        print(f"GPU mode: {args.gpu:d}")
    else:
        device = torch.device("cpu")
        print("CPU mode")

    # モデル作成
        # CNNか全結合モデルか引数で選べるようにしてある
    if args.cnn:
        autoencoder = AE_CNN().to(device)
        print("CNN model")
    else:
        autoencoder = AutoEncoder().to(device)
        print("Fully-connected model")

    # optimizer作成
    optimizer = torch.optim.Adam(autoencoder.parameters(), weight_decay=args.lam)

    # データを読み込み
    train = MNIST(root=".", download=True, train=True,
                  transform=lambda x: np.asarray(x, dtype=np.float32).ravel() / 255)
    test = MNIST(root=".", download=True, train=False,
                 transform=lambda x: np.asarray(x, dtype=np.float32).ravel() / 255)
    # data loaderを作る
    train_loader = DataLoader(train, args.batchsize)

    saver = MNISTSampleSaver(autoencoder, test, 10, args.result, device)
    saver.save_images("before")

    autoencoder.train()
    # 学習ループ
    for i in range(1, args.epoch + 1):
        total_loss = 0
        for x in train_loader:
            x = x.to(device)

            # 再構成損失を計算
            x_recon = autoencoder(x)
            loss = F.mse_loss(x.flatten(start_dim=1), x_recon.flatten(start_dim=1))
            total_loss += loss.item() * len(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch {i:d}: Average train loss = {total_loss/len(train):f}")

    saver.save_images("after")

    # テスト精度を算出
    print("Test phase")

    test_loader = DataLoader(test, args.batchsize)

    autoencoder.eval()
    total_loss = 0
    for x in test_loader:
        x = x.to(device)
        with torch.no_grad():
            x_recon = autoencoder(x)
            loss = F.mse_loss(x.flatten(start_dim=1), x_recon.flatten(start_dim=1))
        total_loss += loss.item() * len(x)
    print(f"Average test loss: {total_loss/len(test):f}")


class MNIST(torchvision.datasets.MNIST):
    def __getitem__(self, i):
        x, _ = super().__getitem__(i)
        return x


class MNISTSampleSaver:
    def __init__(self, model, dataset, num: int, outdir: Path, device):
        self.outdir_path = Path(outdir)
        try:
            self.outdir_path.mkdir(parents=True)
        except FileExistsError:
            pass

        self.model = model
        self.dataset = dataset
        self.device = device
        self.indices = np.random.choice(len(dataset), size=num, replace=False)

        # 選んだ画像を保存
        samples = self._get_samples().reshape(-1, 28, 28) * 255
        samples = samples.astype(np.uint8)
        for i, img in enumerate(samples):
            p = self.outdir_path / f"input_{i:02d}.png"
            cv2.imwrite(str(p), img)

    def save_images(self, kw):
        samples = self._get_samples().reshape(-1, 1, 28, 28)
        samples = torch.from_numpy(samples).to(self.device).float()
        self.model.eval()
        with torch.no_grad():
            sample_out = self.model(samples).detach().cpu().numpy() * 255

        sample_out = sample_out.astype(np.uint8).reshape(-1, 28, 28)

        # 再構成画像を保存
        for i, img in enumerate(sample_out):
            p = self.outdir_path / f"out_{kw}_{i:02d}.png"
            cv2.imwrite(str(p), img)

    def _get_samples(self):
        return np.stack([self.dataset[i] for i in self.indices])


if __name__ == "__main__":
    parse_args()
