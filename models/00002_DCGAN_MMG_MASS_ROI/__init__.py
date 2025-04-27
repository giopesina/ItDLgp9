"""
Author: Basel Alyafi
Master Thesis Project
Erasmus Mundus Joint Master in Medical Imaging and Applications

21062021 & 24092021: adjusted by Richard Osuala (BCN-AIM)
"""
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

# from skimage import io

# custom weights initialization called on netG and netD


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


######################################################################
# Generator
# ~~~~~~~~~


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(
                nz, ngf * 10, kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(ngf * 10),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(
                ngf * 10, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(
                ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(
                ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(
                ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. (ngf) x 64 X 64
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 X 128
        )

    def forward(self, input):
        return self.model(input)

    # a path that can be used for saving the model
    Gpath = "/home/basel/PycharmProjects/DCGAN/models/Generators/"


######################################################################
# Discriminator


class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input size is 128 X 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (nc) x 64 x 64
            nn.Conv2d(
                ndf, ndf * 2, kernel_size=6, stride=2, padding=2, bias=False
            ),  # was 4,2,1
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=6, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=6, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(
                ndf * 8, ndf * 10, kernel_size=6, stride=2, padding=2, bias=False
            ),
            nn.BatchNorm2d(ndf * 10),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 10, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)

    Dpath = "/home/basel/PycharmProjects/DCGAN/models/Discriminators/"


######################################################################


class PSGenerator(nn.Module):
    def __init__(self, L, nz, ngf, nc):
        super(PSGenerator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.L = L

        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(int(ngf / 4)),
            nn.LeakyReLU(),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(int(ngf / 16)),
            nn.LeakyReLU(),
            nn.PixelShuffle(2),
            nn.Tanh(),
        )

    def forward(self, *input):
        return self.model(*input)


def run_generator(
    model, batch_size, save_path, RGB, save_images: bool, input_latent_vector=None
):
    """
    to run a generator to generate images and save them.

    Params
    ------
    model: nn.Module
        the model to run
    batch_size: int
        number of images to generate
    save_path: string
        where to save the generated images
    RGB: bool
        if True images will be saved in RGB format, otherwise grayscale will be used

    Returns
    -------
    void
    """
    # detect if there is a GPU, otherwise use cpu instead
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # model input
    if input_latent_vector is None:
        fixed_noise = torch.randn(batch_size, model.nz, 1, 1, device=device)
    else:
        fixed_noise = torch.from_numpy(input_latent_vector).to(device)

    model.to(device)

    # Testing mode
    mode = model.training
    model.eval()
    model.apply(apply_dropout)

    # generate output samples with generator model
    try:
        with torch.no_grad():
            output = model(fixed_noise).detach().cpu()
    except Exception as e:
        logging.error(f"Error while generating images: {e}")
        raise e

    # post-process images
    post_processed_images = []
    for i in range(batch_size):
        try:
            # rescale intensities from [-1,1] to [0,1]
            img = np.transpose(output[i], [1, 2, 0]) / 2 + 0.5
            # img = np.squeeze(img)
            img = np.array(255 * img).round().astype(np.uint8)

            # if one channel, squeeze to 2d
            if img.shape[2] == 1:
                img = img.squeeze(axis=2)

            # if gray but RGB required
            if RGB and len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            post_processed_images.append(img)
        except Exception as e:
            logging.error(f"Error while post-processing images: {e}")
            raise e

    if save_images:
        # create the path if does not exist.
        if not (os.path.exists(save_path)):
            os.makedirs(save_path, exist_ok=True)
        # save the image
        # io.imsave(save_path + '/{}.png'.format(i), img)
        for j, img in enumerate(post_processed_images):
            cv2.imwrite(save_path + f"/{j}.png", img)
        logging.debug(
            f"Finished generating {batch_size} images. Stored them in {save_path}"
        )
    else:
        logging.debug(
            f"Finished generating {batch_size} images. Returning them now as {type(post_processed_images)}."
        )
        return post_processed_images


def apply_dropout(layer):  # DOC OK
    """
    This function is used to activate dropout layers during training

    Params:
    -------
    layer: torch.nn.Module
        the layer for which the dropout to be activated

    Returns:
    --------
    void
    """
    classname = layer.__class__.__name__
    if classname.find("Dropout") != -1:
        layer.train()


def generate(
    model_file, num_samples, output_path, save_images: bool, input_latent_vector=None
):
    """ This function generates synthetic images of mammography regions of interest """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # the name of the generator (for calcifications we can use 'mass_calcification_gen')
        n_imgs = num_samples

        # create an instance of the generator
        model = Generator(ngpu=1, nz=200, ngf=45, nc=1)
        device_as_string = "cuda" if torch.cuda.is_available() else "cpu"

        # load the pretrained weights
        model.load_state_dict(
            torch.load(model_file, map_location=torch.device(device_as_string))
        )

        # run the trained generator to generate n_imgs images at imgs_path
        return run_generator(
            model=model,
            batch_size=n_imgs,
            save_path=output_path,
            RGB=False,
            save_images=save_images,
            input_latent_vector=input_latent_vector,
        )
    except Exception as e:
        logging.error(
            f"Error while trying to generate {num_samples} images with model {model_file}: {e}"
        )
        raise e
