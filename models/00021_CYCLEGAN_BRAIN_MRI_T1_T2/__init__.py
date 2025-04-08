import torch.nn as nn
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import logging
from pathlib import Path


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# normalizes input data from [-1, 1]
def normalize(x):
    new_x = (2 * (x - torch.min(x)) / (torch.max(x) - torch.min(x))) - 1
    return new_x


def process_one_image(image_path, model, device):
    try:
        # Preprocessing
        base_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Resize((224, 192)),
            normalize  # normalizes image from [-1, 1]
        ])
        image = Image.open(image_path).convert('RGB')  # Load PIL image
        image = base_transform(image)  # Resizing and normalizing

        # Model prediction
        image_gen = torch.squeeze(model(torch.unsqueeze(image.to(device), dim=0)))
        image_gen = (image_gen + 1.0) * 0.5  # Rescale intensity values from [-1, 1] to [0, 1]
        return image_gen

    except Exception as e:
        logging.error(f"Failed to process {image_path}: {e}")
        raise e


def generate(model_file, input_path, output_path, num_samples, save_images=True, gpu_id=0,
             translate_all_images: bool = False, T1_to_T2: bool = True):
    base_path = Path(model_file).parent.absolute()  # Change

    # Check if gpu is available
    device = torch.device("cuda" if (torch.cuda.is_available() and gpu_id is not None and gpu_id != -1) else "cpu")
    if str(device) == 'cuda':
        torch.cuda.set_device(gpu_id)

    try:
        # Initializing model and loading checkpoint
        if T1_to_T2:
            checkpoint = os.path.join(base_path, 'netG_T1toT2_checkpoint.pth.tar')
        else:
            checkpoint = os.path.join(base_path, 'netG_T2toT1_checkpoint.pth.tar')

        logging.debug('Instantiating model...')

        model = Generator(input_nc=1, output_nc=1).to(device)
        pretrained_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(pretrained_dict)
        model = model.eval()

        output_images = []
        logging.debug('Generating images...')

        # if the user inputs a directory of images
        if os.path.isdir(input_path):

            # if a folder is given, the images in that folder are translated until either all images are
            # translated (e.g., if translate_all_images=True) or until <num_samples> images were translated.
            counter = 0
            for file in os.listdir(input_path):
                if file.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    image_path = os.path.join(input_path, file)
                    image_gen = process_one_image(image_path, model, device)
                    if save_images:
                        if not os.path.exists(output_path):
                            os.mkdir(output_path)
                        file_name = Path(file).name
                        plt.imsave(os.path.join(output_path, f"{'T2' if T1_to_T2 else 'T1'}_{file_name}"),
                                   image_gen.detach().cpu().numpy(), cmap='gray')
                        # plt.imsave(os.path.join(output_path,image_path.split('/')[-1]), image_gen.detach().cpu().numpy(), cmap = 'gray')
                    else:
                        output_images.append(image_gen.detach().cpu().numpy())
                    counter = counter + 1
                    if num_samples is not None and counter >= num_samples and not translate_all_images:
                        break
                else:
                    logging.error(
                        'Invalid file format. Allowed formats are ".png", ".jpg", ".jpeg", ".tif", ".tiff" only. Please check again!')

        # if the user inputs a single file
        elif os.path.isfile(input_path):
            if input_path.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                image_gen = process_one_image(input_path, model, device)
                if save_images:
                    if not os.path.exists(output_path):
                        os.mkdir(output_path)
                    file_name = Path(input_path).name
                    plt.imsave(os.path.join(output_path, f"{'T2' if T1_to_T2 else 'T1'}_{file_name}"), image_gen.detach().cpu().numpy(), cmap = 'gray')
                    # plt.imsave(os.path.join(output_path,input_path.split('/')[-1]), image_gen.detach().cpu().numpy(), cmap = 'gray')
                else:
                    output_images.append(image_gen.detach().cpu().numpy())
        else:
            logging.error("Input path is not a valid file")

        if not save_images:
            return output_images

    except Exception as e:
        logging.error(f"Error while trying to generate images with model {model_file}: {e}")
        raise e

