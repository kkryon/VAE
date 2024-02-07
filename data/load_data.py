import torchvision.datasets as datasets
from torch.utils import data
import torchvision.transforms as transforms
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Data:
    def __init__(self):

        self.mnist_train = datasets.MNIST(
            "data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
            ),
        )
        self.mnist_test = datasets.MNIST(
            "data",
            train=True,
            download=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
            ),
        )

    def get_dataloaders(self, train_bs=1024, test_bs=1024):
        train_loader = data.DataLoader(
            self.mnist_train,
            batch_size=train_bs,
            shuffle=True,
            num_workers=5,
        )
        test_loader = data.DataLoader(
            self.mnist_test, batch_size=test_bs, shuffle=False, num_workers=5
        )
        return train_loader, test_loader

    def show_images(self, train_loader):
        examples = enumerate(train_loader)
        _, (example_data, example_targets) = next(examples)
        import matplotlib.pyplot as plt

        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        fig.savefig("data/mnist.png")
        plt.close()
        # print("saved")
