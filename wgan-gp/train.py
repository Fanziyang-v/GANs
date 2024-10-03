import os
from argparse import Namespace, ArgumentParser
import torch
from torch import nn, Tensor, autograd
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from wgan_gp import Generator, Discriminator

# Image processing.
transform_mnist = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))])

transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Device configuration.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def denormalize(x: Tensor) -> Tensor:
    out = (x + 1) / 2
    return out.clamp(0, 1)


def calc_gradient_penalty(real_images: Tensor, fake_images: Tensor, D: Discriminator, args: Namespace) -> Tensor:
    epsilon = torch.rand(args.batch_size, 1).to(device)
    interpolates = (epsilon * real_images + (1 - epsilon) * fake_images).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.ones(args.batch_size, 1).to(device), requires_grad=False)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# mini-batch generator
def generate(data_loader: DataLoader):
    while True:
        for images, _ in data_loader:
            # Reach the last batch without enough images, just break
            if data_loader.batch_size != len(images):
                break
            yield images


def get_args() -> Namespace:
    """Get commandline arguments."""
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for Adam optimizer')
    parser.add_argument('--beta1', type=float, default=0, help='first momentum term for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='second momentum term for Adam optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='size of a mini-batch')
    parser.add_argument('--gen_iters', type=int, default=100000, help='training epochs')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--n_critic', type=int, default=5, help='number of training iterations in discriminator per generator training iteration')
    parser.add_argument('--alpha', type=float, default=10, help='gradient penalty parameter')
    parser.add_argument('--dataset', type=str, default='MNIST', help='training dataset(MNIST | FashionMNIST | CIFAR10)')
    parser.add_argument('--sample_dir', type=str, default='samples', help='directory of image samples')
    parser.add_argument('--interval', type=int, default=1000, help='epoch interval between image samples')
    parser.add_argument('--logdir', type=str, default='runs', help='directory of running log')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='directory for saving model checkpoints')
    parser.add_argument('--seed', type=str, default=10213, help='random seed')
    return parser.parse_args()


def setup(args: Namespace) -> None:
    torch.manual_seed(args.seed)
    # Create directory if not exists.
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)


def get_data_loader(args: Namespace) -> DataLoader:
    """Get data loader."""
    if args.dataset == 'MNIST':
        data = datasets.MNIST(root='../data', train=True, download=True, transform=transform_mnist)
    elif args.dataset == 'FashionMNIST':
        data = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform_mnist)
    elif args.dataset == 'CIFAR10':
        data = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_cifar)
    else:
        raise ValueError(f'Unkown dataset: {args.dataset}, support dataset: MNIST | FashionMNIST | CIFAR10')
    return DataLoader(dataset=data, batch_size=args.batch_size, num_workers=4, shuffle=True)


def train(args: Namespace, 
          G: Generator, D: Discriminator, 
          data_loader: DataLoader) -> None:
    """Train Generator and Discriminator.

    Args:
        args(Namespace): arguments.
        G(Generator): Generator in GAN.
        D(Discriminator): Discriminator in GAN.
    """
    gen = generate(data_loader)
    writer = SummaryWriter(args.logdir)

    # generate fixed noise for sampling.
    fixed_noise = torch.rand(64, args.latent_dim).to(device)

    # Loss and optimizer.
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    for i in range(args.gen_iters):
        total_d_loss = 0
        for _ in range(args.n_critic):
            real_images = next(gen)
            real_images: Tensor = real_images.to(device)
            z = torch.rand(args.batch_size, args.latent_dim).to(device)
            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #

            # Forward pass
            
            fake_images = G(z)
            real_score: Tensor = D(real_images)
            fake_score: Tensor = D(fake_images)
            gradient_penalty = calc_gradient_penalty(real_images, fake_images, D, args)
            d_loss: Tensor = (fake_score - real_score).mean() + args.alpha * gradient_penalty
            
            # Backward pass
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
            total_d_loss += d_loss
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #
        
        # Forward pass
        z = torch.rand(args.batch_size, args.latent_dim).to(device)
        fake_score: Tensor = D(G(z))
        g_loss = (-fake_score).mean()
        
        # Backward pass
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
        
        print(f'''
=====================================
Step: [{i + 1}/{args.gen_iters}]
Discriminator Loss: {total_d_loss / args.n_critic:.4f}
Generator Loss: {g_loss:.4f}
=====================================''')
        # Log Discriminator and Generator loss.
        writer.add_scalar('Discriminator Loss', total_d_loss / args.n_critic, i + 1)
        writer.add_scalar('Generator Loss', g_loss, i + 1)
        if (i + 1) % args.interval == 0:
            fake_images: Tensor = G(fixed_noise)
            img_grid = make_grid(denormalize(fake_images), nrow=8, padding=2)
            writer.add_image('Fake Images', img_grid, (i + 1) // args.interval)
            save_image(img_grid, os.path.join(args.sample_dir, f'fake_images_{(i + 1) // args.interval}.png'))
    # Save the model checkpoints.
    torch.save(G.state_dict(), os.path.join(args.ckpt_dir, 'G.ckpt'))
    torch.save(D.state_dict(), os.path.join(args.ckpt_dir, 'D.ckpt'))


def main() -> None:
    args = get_args()
    setup(args)
    C = 1 if args.dataset in ('MNIST', 'FashionMNIST') else 3
    data_loader = get_data_loader(args)
    # Generator and Discrminator.
    G = Generator(num_channels=C, latent_dim=args.latent_dim).to(device)
    D = Discriminator(num_channels=C).to(device)
    train(args, G, D, data_loader)


if __name__ == '__main__':
    main()
