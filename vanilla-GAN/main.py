import os
import argparse
import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from gan import Generator, Discriminator

# Transform for grayscale images(MNIST | FashionMNIST)
transform1c = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))])

# Transform for color images(CIFAR-10)
transform3c = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

def denorm(x: torch.Tensor) -> torch.Tensor:
    out = (x + 1) / 2
    return out.clamp(0, 1)


# Device configuration.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(options: argparse.Namespace) -> None:
    # Setting hyper parameters.
    LEARNING_RATE = options.lr
    BATCH_SIZE = options.batch_size
    NUM_EPOCHS = options.num_epochs
    LATENT_SIZE = options.latent_size
    C, H, W = (1, 28, 28) if options.dataset in ('MNIST', 'FashionMNIST') else (3, 32, 32)
    IMAGE_SIZE = C * H * W
    SAMPLES_DIR = options.samples_dir
    INTERVAL = options.interval
    CHECKPOINTS_DIR = options.ckpt_dir
    # Dataset and data loader.
    if options.dataset == 'MNIST':
        data = datasets.MNIST(root='../data', train=True, download=True, transform=transform1c)
    elif options.dataset == 'FashionMNIST':
        data = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform1c)
    elif options.dataset == 'CIFAR10':
        data = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform3c)
    else:
        raise ValueError(f'Unkwon Dataset: {options.dataset}, Support dataset: MNIST(Default) | FashionMNIST | CIFAR10')
    data_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

    # Generator and Discriminator.
    D = Discriminator(IMAGE_SIZE).to(device)
    G = Generator(IMAGE_SIZE, LATENT_SIZE).to(device)
    # Running on multiple GPU or CPU.
    D = nn.DataParallel(D)
    G = nn.DataParallel(G)
    # Load pre-train model if specified checkpoint path.
    if options.d_ckpt:
        D.load_state_dict(torch.load(options.d_ckpt, weights_only=True))
    if options.g_ckpt:
        G.load_state_dict(torch.load(options.g_ckpt, weights_only=True))
    # Loss and optimizer.
    criterion = nn.BCELoss().to(device)
    if options.optimizer == 'adam':
        G_optimizer = torch.optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        D_optimizer = torch.optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    elif options.optimizer == 'sgd':
        G_optimizer = torch.optim.SGD(G.parameters(), lr=LEARNING_RATE, momentum=0.9)
        D_optimizer = torch.optim.SGD(D.parameters(), lr=LEARNING_RATE, momentum=0.9)
    elif options.optimizer == 'rmsprop':
        G_optimizer = torch.optim.RMSprop(G.parameters(), lr=LEARNING_RATE)
        D_optimizer = torch.optim.RMSprop(D.parameters(), lr=LEARNING_RATE)
    else:
        raise ValueError(f'Unkown Optimizer: {options.optimizer}, Support optimizer: adam(Default) | sgd | rmsprop')
    
    # Create a directory if not exists
    if not os.path.exists(SAMPLES_DIR):
        os.makedirs(SAMPLES_DIR)
    if not os.path.exists(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)
    # Start training.
    for epoch in range(NUM_EPOCHS):
        for images, _ in data_loader:
            images: torch.Tensor = images.to(device)
            images = images.view(images.size(0), -1)
            real_labels = torch.ones(images.size(0), 1).to(device)
            fake_labels = torch.zeros(images.size(0), 1).to(device)
            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #
            
            # Forward pass
            outputs: torch.Tensor = D(images)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            z = torch.randn(images.size(0), LATENT_SIZE).to(device)
            fake_images: torch.Tensor = G(z)
            outputs: torch.Tensor = D(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            # Backprop and optimize
            d_loss: torch.Tensor = d_loss_real + d_loss_fake
            D_optimizer.zero_grad()
            d_loss.backward()
            D_optimizer.step()

            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #
            
            # Forward pass
            z = torch.randn(images.size(0), LATENT_SIZE).to(device)
            fake_images: torch.Tensor = G(z)
            outputs: torch.Tensor = D(fake_images)

            # Backprop and optimize
            D_optimizer.zero_grad()
            G_optimizer.zero_grad()
            g_loss: torch.Tensor = criterion(outputs, real_labels)
            g_loss.backward()
            G_optimizer.step()
        # Saving real images at first epoch.
        if (epoch + 1) == 1:
            images = images.view(-1, C, H, W)
            save_image(denorm(images), os.path.join(SAMPLES_DIR, 'real_images.png'))
        # Save sampled images
        if (epoch + 1) % INTERVAL == 0:
            fake_images = fake_images.view(-1, C, H, W)
            save_image(denorm(fake_images), os.path.join(SAMPLES_DIR, f'fake-images-{(epoch + 1) // INTERVAL}.png'))
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}, D(x): {real_score.mean().item()}, D(G(z)): {fake_score.mean().item()}')

    # Save the model checkpoints 
    torch.save(G.state_dict(), os.path.join(CHECKPOINTS_DIR, 'G.ckpt'))
    torch.save(D.state_dict(), os.path.join(CHECKPOINTS_DIR, 'D.ckpt'))


def inference(options: argparse.Namespace) -> None:
    if not options.g_ckpt:
        raise ValueError(f'Use --g_ckpt flag to specify generator checkpoint path in inference mode')
    if options.dataset not in ('MNIST', 'FashionMNIST', 'CIFAR10'):
        raise ValueError(f'Unkwon Dataset: {options.dataset}, Support dataset: MNIST(Default) | FashionMNIST | CIFAR10')
    # Setting hyper parameters.
    LATENT_SIZE = options.latent_size
    C, H, W = (1, 28, 28) if options.dataset in ('MNIST', 'FashionMNIST') else (3, 32, 32)
    IMAGE_SIZE = C * H * W
    BATCH_SIZE = options.batch_size
    SAMPLES_DIR = options.samples_dir
    G = Generator(IMAGE_SIZE, LATENT_SIZE).to(device)
    G = nn.DataParallel(G)
    z = torch.rand(BATCH_SIZE, LATENT_SIZE)
    fake_images = G(z)
    save_image(denorm(fake_images), os.path.join(SAMPLES_DIR, 'fake_images.png'))


# Get Hyper parameters.
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset: MINIST | FashionMNIST | CIFAR10')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--latent_size', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer: adam | sgd | rmsprop')
    parser.add_argument('--samples_dir', type=str, default='samples', help='directory of image samples')
    parser.add_argument('--interval', type=int, default=1, help='epoch interval between image samples')
    parser.add_argument('--g_ckpt', type=str, default='', help='generator checkpoint path')
    parser.add_argument('--d_ckpt', type=str, default='', help='discriminator checkpoint path')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='directory for saving model checkpoints after training')
    parser.add_argument('--inference', help='inference mode')
    options = parser.parse_args()
    print(options)
    return options


def main() -> None:
    # Get command line parameters.
    params = get_args()
    if params.inference:
        inference(params)
    else:
        train(params)

if __name__ == '__main__':
    main()
