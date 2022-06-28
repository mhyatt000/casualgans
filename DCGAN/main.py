import warnings

from d2l import torch as d2l
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
from tqdm import tqdm

warnings.filterwarnings("ignore")


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""

    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


class G_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2, padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, strides, padding, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))


class D_block(nn.Module):
    def __init__(
        self, out_channels, in_channels=3, kernel_size=4, strides=2, padding=1, alpha=0.2, **kwargs
    ):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, strides, padding, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))


def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim, device='cpu'):

    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)

    net_D, net_G = net_D.to(device), net_G.to(device)
    trainer_hp = {'lr': lr, 'betas': [0.5,0.999]}
    trainer_D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
    trainer_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs], nrows=2, figsize=(5, 5), legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)

    # for epoch in tqdm(range(1, num_epochs + 1)):
    for epoch in range(1, num_epochs + 1):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples

        for X, _ in tqdm(data_iter):
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.to(device), Z.to(device)
            metric.add(d2l.update_D(X, Z, net_D, net_G, loss, trainer_D),
                       d2l.update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)

        # Show generated examples
        Z = torch.normal(0, 1, size=(21, latent_dim, 1, 1), device=device)
        # Normalize the synthetic data to N(0, 1)
        fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
        imgs = torch.cat( [torch.cat([
                fake_x[i * 7 + j].cpu().detach() for j in range(7)], dim=1)
             for i in range(len(fake_x)//7)], dim=0
        )

        animator.axes[1].cla()
        animator.axes[1].imshow(imgs)
        # Show the losses
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec on {str(device)}')


def main():
    """main method"""

    data_dir = "/Users/matthewhyatt/cs/.datasets/pokemon"
    pokemon = torchvision.datasets.ImageFolder(data_dir)

    batch_size = 256
    transformer = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5),
        ]
    )

    pokemon.transform = transformer
    data_iter= torch.utils.data.DataLoader(pokemon, batch_size=batch_size, shuffle=True)

    # for x,y in data_iter:
    # imgs = x[0:20, :, :, :].permute(0, 2, 3, 1) / 2 + 0.5
    # show_images(imgs, num_rows=4, num_cols=5)
    # plt.show()
    # break

    n_G = 64
    net_G = nn.Sequential(
        G_block( in_channels=100, out_channels=n_G * 8, strides=1, padding=0),  
        G_block(in_channels=n_G * 8, out_channels=n_G * 4),  
        G_block(in_channels=n_G * 4, out_channels=n_G * 2),  
        G_block(in_channels=n_G * 2, out_channels=n_G),  
        nn.ConvTranspose2d(
            in_channels=n_G, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False
        ),
        nn.Tanh(),
    )  

    n_D = 64
    net_D = nn.Sequential(
        D_block(n_D),  
        D_block(in_channels=n_D, out_channels=n_D*2),  
        D_block(in_channels=n_D*2, out_channels=n_D*4),  
        D_block(in_channels=n_D*4, out_channels=n_D*8),  
        nn.Conv2d(in_channels=n_D*8, out_channels=1, kernel_size=4, bias=False)
    )  

    latent_dim, lr, num_epochs = 100, 0.005, 30
    train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)

if __name__ == "__main__":
    main()
