import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from math import sqrt
import numpy as np
import os
from tqdm import tqdm
import datetime
import pickle
import datetime
import time




# Hyperparameters
DATASET                 = "/app/Training/data/flickr/"
DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu" # run on gpu if possible
EPOCHS                  = 151
LEARNING_RATE           = 1e-3 # 0.001``
BATCH_SIZE              = 12
RESOLUTION              = "256x256"
LOG_RESOLUTION          = 8 # 7 for 128*128 | trying to generate 128*128 images, and 2^7 = 128
                            # 8 = 256
                            # 9 = 512
                            # 10 = 1024 (runs out of memory)
Z_DIM                   = 512
W_DIM                   = 512
LAMBDA_GP               = 10 # This loss contains a parameter name λ and it's common to set λ = 10
LOAD_CHECKPOINT         = False
LOAD_EPOCH              = 140
CHECKPOINT_PATH         = f"/app/Training/output/{RESOLUTION}/epoch{LOAD_EPOCH}/facecraft_{RESOLUTION}_{LOAD_EPOCH}.pth"
DATE                    = datetime.datetime.now().strftime("%Y-%m-%d")
OUTPUT_FOLDER           = f"/app/Training/output/{RESOLUTION}/"

 
# Get data loader
def get_loader(): 
    # Apply some transformation to the images
    transform = transforms.Compose(
        [
            # resize the images to the resolution that we want
            transforms.Resize((2 ** LOG_RESOLUTION, 2 ** LOG_RESOLUTION)),
            # convert them to tensors
            transforms.ToTensor(),
            #  apply some augmentation
            transforms.RandomHorizontalFlip(p=0.5),
            # normalize them to be all the pixels ranging from -1 to 1
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ]
    )
    # Prepare the dataset by using ImageFolder
    dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    # Create mini-batch sizes
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    # return the loader
    return loader

# Models implementation
# Implement the StyleGAN2 networks with the key attributions from the paper
# The key points:
#   Noise Mapping Network
#   Weight demodulation (Instead of Adaptive Instance Normalization (AdaIN))
#   Skip connections (Instead of progressive growing)
#   Perceptual path length normalization

# Noise Mapping Network
#   create the MappingNetwork class which will be inherited from nn.Module
class MappingNetwork(nn.Module):
    # send z_dim and w_din
    def __init__(self, z_dim, w_dim):
        super().__init__()
        # define the network mapping containing eight of EqualizedLinear
        self.mapping = nn.Sequential(
            # equalizes the learning rate
            EqualizedLinear(z_dim, 512),
            # activation function
            nn.ReLU(),
            EqualizedLinear(512, 512),
            nn.ReLU(),
            EqualizedLinear(512, 512),
            nn.ReLU(),
            EqualizedLinear(512, 512),
            nn.ReLU(),
            EqualizedLinear(512, 512),
            nn.ReLU(),
            EqualizedLinear(512, 512),
            nn.ReLU(),
            EqualizedLinear(512, 512),
            nn.ReLU(),
            EqualizedLinear(512, w_dim)
        )

    def forward(self, x):
        # initialize z_dim using pixel norm
        x = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)  # for PixelNorm 
        # return the network mapping
        return self.mapping(x)
    
# Generator
class Generator(nn.Module):

    def __init__(self, log_resolution, W_DIM, n_features = 32, max_features = 256):
        # log_resolution which is the log2​ of image resolution
        # W_DIM which s the dimensionality of w
        # n_featurese which is the number of features in the convolution layer at the highest resolution (final block)
        # max_features which is the maximum number of features in any generator block
        super().__init__()

        # calculate the number of features for each block
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        # get the number of generator blocks
        self.n_blocks = len(features)

        # initialize the trainable 4x4 constant
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        # first style block for 4×4 resolution
        self.style_block = StyleBlock(W_DIM, features[0], features[0])
        # layer to get RGB
        self.to_rgb = ToRGB(W_DIM, features[0])

        # the generator blocks
        blocks = [GeneratorBlock(W_DIM, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, w, input_noise):
        # w for each generator block | w.shape[n_blocks, batch_size, W-dim]
        # input_noise is the noise for each block | list of pairs of noise tensors, each block (except the initial) has two noise inputs after each convolution layer

        # get the batch size
        batch_size = w.shape[1]

        # expand the learned constant to match the batch size
        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        # run it into the first style block
        x = self.style_block(x, w[0], input_noise[0][1])
        # get the RGB image
        rgb = self.to_rgb(x, w[0])

        # run it again into the rest of the generator blocks after upsampling
        for i in range(1, self.n_blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear") + rgb_new

        # return the last RGB image with tanh as an activation function
        # tanh because we want the pixels of the generated image to range between 1 and -1
        return torch.tanh(rgb)

# Generator Block
class GeneratorBlock(nn.Module):

    def __init__(self, W_DIM, in_features, out_features):
        # W_DIM is the dimensionality of w
        # in_features is the number of features in the input feature map
        # out_features is the number of features in the output feature map
        super().__init__()

        # initialize two style blocks
        self.style_block1 = StyleBlock(W_DIM, in_features, out_features)
        self.style_block2 = StyleBlock(W_DIM, out_features, out_features)

        # toRGB layer
        self.to_rgb = ToRGB(W_DIM, out_features)

    def forward(self, x, w, noise):
        # x is the input feature map of the shape [batch_size, in_features, height, width]
        # w with the shape [batch_size, W_DIM]
        # noise is a tuple of two noise tensors of shape [batch_size, 1, height, width]

        # run x into the two style blocks
        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])

        # get the RGB image using the layer toRGB
        rgb = self.to_rgb(x, w)

        # return x and the RGB image
        return x, rgb
    
# Style Block
class StyleBlock(nn.Module):

    def __init__(self, W_DIM, in_features, out_features):
        # W_DIM is the dimensionality of w
        # in_features is the number of features in the input feature map
        # out_features is the number of features in the output feature map
        super().__init__()
 
        # initialize to_style by the style vector from w with an equalized learning rate linear layer
        self.to_style = EqualizedLinear(W_DIM, in_features, bias=1.0)
        # weight modulated convolution layer
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        # noise scale
        self.scale_noise = nn.Parameter(torch.zeros(1))
        # bias
        self.bias = nn.Parameter(torch.zeros(out_features))
        # activation function
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w, noise):
        # x is the input feature map of the shape [batch_size, in_features, height, width]
        # w with the shape [batch_size, W_DIM]
        # noise is a tuple of two noise tensors of shape [batch_size, 1, height, width]

        # get the style vector s
        s = self.to_style(w)
        # run x and s into the weight modulated convolution
        x = self.conv(x, s)
        # scale and add noise
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        # add bias and evaluate the activation function
        return self.activation(x + self.bias[None, :, None, None])

# To RGB
class ToRGB(nn.Module):

    def __init__(self, W_DIM, features):
        # W_DIM is the dimensionality of w

        super().__init__()
        # initialize to_style by the style vector that we get from w
        self.to_style = EqualizedLinear(W_DIM, features, bias=1.0)

        # weight modulated convolution layer
        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        # bias
        self.bias = nn.Parameter(torch.zeros(3))
        # activation function
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w):
        # x is the input feature map of the shape [batch_size, in_features, height, width]
        # w with the shape [batch_size, W_DIM]

        # get the style vector style
        style = self.to_style(w)
        # run x and style into the weight modulated convolution
        x = self.conv(x, style)
        # add bias and evaluate the activation function
        return self.activation(x + self.bias[None, :, None, None])
    
# Convolution with Weight Modulation and Demodulation
#    scales the convolution weights by the style vector and demodulates it by normalizing it
class Conv2dWeightModulate(nn.Module):

    def __init__(self, in_features, out_features, kernel_size,
                 demodulate = True, eps = 1e-8):
        # in_features is the number of features in the input feature map
        # out_features is the number of features in the output feature map
        # demodulates is a flag to normalize weights by its standard deviation
        # eps is the ϵ for normalizing

        super().__init__()
        # initialize the number of output features
        self.out_features = out_features
        # demodulate
        self.demodulate = demodulate
        # padding size
        self.padding = (kernel_size - 1) // 2
        # Weights parameter with equalized learning rate
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        # eps
        self.eps = eps

    def forward(self, x, s):
        # x is the input feature map of the shape [batch_size, in_features, height, width]
        # s is a style-based scaling tensor

        # get the batch size, height, and width from x
        b, _, h, w = x.shape
        # reshape the scales
        s = s[:, None, :, None, None]
        # get the learning rate equalized weights
        weights = self.weight()[None, :, :, :, :]
        # modulate x and s
        weights = weights * s

        # demodulate them if demodulates is True
        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)
        # return x
        return x.reshape(-1, self.out_features, h, w)
    
# Discriminator
#   transforms the image with the resolution 2LOG_RESOLUTION by 2LOG_RESOLUTION  to a feature map of the same resolution
#   runs it through a series of blocks with residual connections
#   resolution is down-sampled by 2× at each block while doubling the number of features
class Discriminator(nn.Module):

    def __init__(self, log_resolution, n_features = 64, max_features = 256):

        super().__init__()
        # calculate the number of features for each block
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]

        # initialize a layer to convert the RGB image to a feature map with n_features number of features
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )
        # number of discriminator blocks
        n_blocks = len(features) - 1
        # discriminator blocks
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        # number of features after adding the map of the standard deviation
        self.blocks = nn.Sequential(*blocks)

        final_features = features[-1] + 1
        # final 3×3 convolution layer
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        # final linear layer to get the classification
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    # discriminator will get information about the variation in the batch/image
    def minibatch_std(self, x):
        # take the std for each example we repeat it for a single channel
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        # concatenate it with the image
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x):
        # x is the input feature map of the shape [batch_size, in_features, height, width]

        # run x through the 
        # from_RGB layer
        x = self.from_rgb(x)
        # discriminator blocks
        x = self.blocks(x)
        # minibatch_std
        x = self.minibatch_std(x)
        # 3×3 convolution
        x = self.conv(x)
        # flatten
        x = x.reshape(x.shape[0], -1)
        # classification score
        return self.final(x)
    
# Discriminator Block
#   architecture that consists of two 3×3 convolutions with a residual connection
class DiscriminatorBlock(nn.Module):

    def __init__(self, in_features, out_features):
        # in_features is the number of features in the input feature map
        # out_features is the number of features in the output feature map

        super().__init__()
        # initialize the residual block that contains down-sampling
        # initialize a 1×1 convolution layer for the residual connection
        self.residual = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), # down sampling using avg pool
                                      EqualizedConv2d(in_features, out_features, kernel_size=1))
        # block layer that contains two 3×3 convolutions with Leaky Rely as activation function
        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        # down_sample layer using AvgPool2d
        self.down_sample = nn.AvgPool2d(
            kernel_size=2, stride=2
        )  # down sampling using avg pool

        # scale factor that will be used after adding the residual
        self.scale = 1 / sqrt(2)

    def forward(self, x):
        # x is the input feature map of the shape [batch_size, in_features, height, width]
        
        # run x throw the residual connection to get a variable with the name residual
        residual = self.residual(x)

        #  run x throw the convolutions
        x = self.block(x)
        # downsample
        x = self.down_sample(x)
        # return after add the residual and scale
        return (x + residual) * self.scale
    
# Learning-rate Equalized Linear Layer
#   equalize the learning rate for a linear layer
class EqualizedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias = 0.):
        # in_features is the number of features in the input feature map
        # out_features is the number of features in the output feature map

        super().__init__()
        # initialize the weight
        self.weight = EqualizedWeight([out_features, in_features])
        # initialize the bias
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        # x is the input feature map of the shape [batch_size, in_features, height, width]

        # return the linear transformation of x, weight, and bias
        return F.linear(x, self.weight(), bias=self.bias)

# Learning-rate Equalized 2D Convolution Layer
#   equalize the learning rate for a convolution layer
class EqualizedConv2d(nn.Module):

    def __init__(self, in_features, out_features,
                 kernel_size, padding = 0):
        # in_features is the number of features in the input feature map
        # out_features is the number of features in the output feature map

        super().__init__()
        # initialize the padding
        self.padding = padding
        # initialize weight by a class EqualizedWeight
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        # initialize bias
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        # x is the input feature map of the shape [batch_size, in_features, height, width]

        # return the convolution of x, weight, bias, and padding
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)

# Learning-rate Equalized Weights Parameter
#  Instead of initializing weights at N(0,c) they initialize weights to N(0,1) and then multiply them by c when using it
class EqualizedWeight(nn.Module):

    def __init__(self, shape):
        # shape of the weight parameter
        super().__init__()
        # initialize the constant c
        self.c = 1 / sqrt(np.prod(shape[1:]))
        # initialize weights with N(0,1)
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        # multiply weights by c and return
        return self.weight * self.c

# Perceptual path length normalization
#    encourages a fixed-size step in w to result in a fixed-magnitude change in the image
class PathLengthPenalty(nn.Module):

    def __init__(self, beta):
        # beta is the constant β used to calculate the exponential moving average a
        super().__init__()
        # Initialize beta
        self.beta = beta
        # steps by the number of steps calculated N
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)
        # exp_sum_a by the exponential sum of (J_w)^T (y)
        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w, x):
        # w of shape [batch_size, W_DIM]
        # x is generated images of shape [batch_size, 3, height, width]

        # get the device 
        device = x.device
        # get number of pixels
        image_size = x.shape[2] * x.shape[3]
        y = torch.randn(x.shape, device=device)

        output = (x * y).sum() / sqrt(image_size)
        sqrt(image_size)

        # # calculate the equations
        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)

        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:

            a = self.exp_sum_a / (1 - self.beta ** self.steps)

            loss = torch.mean((norm - a) ** 2)
        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        # update exponential sum
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        # increment N
        self.steps.add_(1.)
        # return the penalty
        return loss
    
# gradient_penalty
#    for WGAN-GP loss
def gradient_penalty(critic, real, fake,device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)
 
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

# Sample W
#   samples Z randomly and gets W from the mapping network
def get_w(batch_size):

    z = torch.randn(batch_size, W_DIM).to(DEVICE)
    w = mapping_network(z)
    return w[None, :, :].expand(LOG_RESOLUTION, -1, -1)

# Generate noise
#   generates noise for each generator block
def get_noise(batch_size):
    
        noise = []
        resolution = 4

        for i in range(LOG_RESOLUTION):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)

            noise.append((n1, n2))

            resolution *= 2

        return noise

# goal of this function is to generate n fake images and save them as a result for each epoch
def generate_examples(gen, epoch, n=5):
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    gen.eval()
    for i in range(n):
        with torch.no_grad():
            w = get_w(1)
            noise = get_noise(1)
            img = gen(w, noise)
            if not os.path.exists(f'{OUTPUT_FOLDER}epoch{epoch}'):
                os.makedirs(f'{OUTPUT_FOLDER}epoch{epoch}', exist_ok=True)
            save_image(img*0.5+0.5, f'{OUTPUT_FOLDER}epoch{epoch}/img_{i}.png')
    gen.train()

# Training
#   train our StyleGAN2
def train_fn(
    critic, # discriminator/critic
    gen, # generator
    path_length_penalty, # will use every 16 epochs
    loader,
    # optimizers for the networks
    opt_critic,
    opt_gen,
    opt_mapping_network,
):
    loop = tqdm(loader, leave=True)
    # looping over all the mini-batch sizes created with DataLoader
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        # take just the images
        cur_batch_size = real.shape[0]

        # how much the critic can distinguish between real and fake images
        w     = get_w(cur_batch_size)
        noise = get_noise(cur_batch_size)
        with torch.cuda.amp.autocast():
            fake = gen(w, noise)
            critic_fake = critic(fake.detach())
            
            critic_real = critic(real)
            gp = gradient_penalty(critic, real, fake, device=DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()
        # maximize E(critic(fake))
        gen_fake = critic(fake)
        loss_gen = -torch.mean(gen_fake)
        # add to this function a perceptual path length every 16 epochs
        if batch_idx % 16 == 0:
            plp = path_length_penalty(w, fake)
            if not torch.isnan(plp):
                loss_gen = loss_gen + plp
        # update the loop
        mapping_network.zero_grad()
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        opt_mapping_network.step()

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )

def export_model_to_onnx(model, input_shape, epoch, w_shape, directory=f"{OUTPUT_FOLDER}", provided_filename="model_test"):
    model.eval()  # Ensure the model is in eval mode for export
    os.makedirs(directory, exist_ok=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"{provided_filename}_{date}_epoch_{epoch}.onnx"
    export_path = os.path.join(directory, filename)
    
    # Generate dummy inputs that match the expected input sizes
    dummy_img_input = torch.randn(input_shape, device=DEVICE)
    dummy_w = torch.randn(w_shape, device=DEVICE)  

    # Export the model
    torch.onnx.export(model, (dummy_img_input, dummy_w), export_path)  # Pass both inputs to the model
    print(f"Model exported to ONNX format at {export_path}")

def save_checkpoint(gen, critic, mapping_network, opt_gen, opt_critic, opt_mapping_network, epoch, directory=f"{OUTPUT_FOLDER}", provided_filename="facecraft"):
    os.makedirs(directory, exist_ok=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    directory = f"{OUTPUT_FOLDER}epoch{epoch}/"
    filename = f"{provided_filename}_{RESOLUTION}_{epoch}.pth"
    path = os.path.join(directory, filename)
    
    # Saving state dictionaries
    torch.save({
        'epoch': epoch,
        'gen_state_dict': gen.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'mapping_network_state_dict': mapping_network.state_dict(),
        'opt_gen_state_dict': opt_gen.state_dict(),
        'opt_critic_state_dict': opt_critic.state_dict(),
        'opt_mapping_network_state_dict': opt_mapping_network.state_dict()
    }, path)
    
    # # Saving entire models
    # torch.save(gen, os.path.join(directory, f"{provided_filename}_gen_full_{date}_epoch_{epoch}_T1.pt"))
    # torch.save(critic, os.path.join(directory, f"{provided_filename}_critic_full_{date}_epoch_{epoch}_T1.pt"))
    # torch.save(mapping_network, os.path.join(directory, f"{provided_filename}_mapping_network_full_{date}_epoch_{epoch}_T1.pt"))
    # torch.save(gen.state_dict(), os.path.join(directory, f"{provided_filename}_model_weights_{date}_epoch_{epoch}_T1.pt"))

    print(f"Checkpoint and full models saved to {directory}")

def load_checkpoint(model, optimizer, path, component):
    """Load model and optimizer states from a checkpoint file."""
    checkpoint = torch.load(path)
    model_state_dict = checkpoint.get(f'{component}_state_dict')
    optimizer_state_dict = checkpoint.get(f'opt_{component}_state_dict')
    if model_state_dict and optimizer_state_dict:
        model.load_state_dict(model_state_dict)
        try:
            optimizer.load_state_dict(optimizer_state_dict)
        except ValueError as e:
            print("Error loading optimizer state dict:", e)
            # Handle partial loading here if necessary
            # optimizer.load_state_dict(optimizer_state_dict, strict=False)
        start_epoch = checkpoint['epoch']
        return model, optimizer, start_epoch
    else:
        raise KeyError(f"Missing required keys in checkpoint: {path}")

def export_checkpoint_to_pickle(checkpoint_path, directory=f"{OUTPUT_FOLDER}", provided_filename="facecraft"):
    # Load the existing checkpoint
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint['epoch']
    directory = f"{OUTPUT_FOLDER}epoch{epoch}/"
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Define the filename and path
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"{provided_filename}_epoch_{epoch}_pickle.pkl"
    path = os.path.join(directory, filename)
    
    # Use pickle to serialize the checkpoint dictionary
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved as a pickle file at {path}")


if __name__ == '__main__':

        torch.cuda.empty_cache()
        #  initialize the loader
        loader              = get_loader()

        # initialize the networks
        gen                 = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)
        critic              = Discriminator(LOG_RESOLUTION).to(DEVICE)
        mapping_network     = MappingNetwork(Z_DIM, W_DIM).to(DEVICE)
        path_length_penalty = PathLengthPenalty(0.99).to(DEVICE)

        # initialize the optimizers
        opt_gen             = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
        opt_critic          = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
        opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

        # make the networks in the training mode
        gen.train()
        critic.train()
        mapping_network.train()

        start_epoch = 0
        

        if LOAD_CHECKPOINT:
            path = CHECKPOINT_PATH
            gen, opt_gen, start_epoch = load_checkpoint(gen, opt_gen, path, 'gen')
            critic, opt_critic, _ = load_checkpoint(critic, opt_critic, path, 'critic')
            mapping_network, opt_mapping_network, _ = load_checkpoint(mapping_network, opt_mapping_network, path, 'mapping_network')
            start_epoch += 1
            print(f'Loaded checkpoint {path} at epoch {start_epoch}. Continuing training for {EPOCHS - start_epoch} more epochs.')
        else:
            print('No checkpoint loaded')

        # train the networks using the training loop
        save_count = 0
        max_count = 1
        for epoch in range(start_epoch, EPOCHS):
            train_fn(
                critic,
                gen,
                path_length_penalty,
                loader,
                opt_critic,
                opt_gen,
                opt_mapping_network,
            )
            save_count += 1
            
            print(f"Finished Training Epoch {epoch}.")
            if save_count >= max_count or epoch == EPOCHS - 1:
                save_count = 0
                generate_examples(gen, epoch)
                save_checkpoint(gen, critic, mapping_network, opt_gen, opt_critic, opt_mapping_network, epoch)
                torch.cuda.empty_cache()
            else: print(f"{save_count}/{max_count} more until saving.")
    
