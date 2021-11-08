# -*- coding: utf-8 -*-
import os

import numpy as np
import progressbar
import skimage.io as io
import sklearn.model_selection as sk
import torch
import torch.nn as nn
import torch.optim as optim

from wnet.models import wnet
from wnet.utils import utils, data, soft_n_cut_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# widget list for the progress bar
widgets = [
    " [",
    progressbar.Timer(),
    "] ",
    progressbar.Bar(),
    " (",
    progressbar.ETA(),
    ") ",
]

BASE_PATH = "/home/edgar/Documents/Datasets/JB/new_images/"
SAVE_PATH = "saved_models/net.pth"
LOSS = np.inf


def save_model(net, loss):
    global LOSS
    if loss < LOSS:
        LOSS = loss
        torch.save(net.state_dict(), SAVE_PATH)


def get_datasets(path_img, config):
    img_path_list = utils.list_files_path(path_img)
    img_path_list = utils.shuffle_list(img_path_list)
    # not good if we need to do metrics
    img_train, img_val = sk.train_test_split(
        img_path_list, test_size=0.2, random_state=42
    )
    dataset_train = data.Unsupervised_dataset(
        config.batch_size, config.size, img_train
    )
    dataset_val = data.Unsupervised_dataset(config.batch_size, config.size, img_val)
    return dataset_train, dataset_val


def train(path_imgs, config, epochs=5):
    net = wnet.Wnet_attention(filters=config.filters, drop_r=config.drop_r).cuda()
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    n_cut_loss = soft_n_cut_loss.NCutLoss2D()
    recons_loss = nn.MSELoss()
    #  get dataset
    dataset_train, dataset_val = get_datasets(path_imgs, config)
    # epoch_enc_loss = []
    # epoch_recons_loss = []

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, verbose=True
    )
    for epoch in range(epochs):
        _enc_loss = []
        _recons_loss = []
        utils.print_gre("Epoch {}/{}".format(epoch + 1, epochs))
        for step in ["Train", "Test"]:
            with progressbar.ProgressBar(  # todo: refactor loop, pas bo
                    max_value=len(dataset_train), widgets=widgets
            ) as bar:
                net.train()
                for i in range(len(dataset_train)):  # boucle inf si on ne fait pas comme ça
                    bar.update(i)
                    imgs = dataset_train[i].cuda()
                    if step == "Train":
                        optimizer.zero_grad()  # zero the gradient buffers
                    mask, att = net.forward_enc(imgs)  # return seg and attention map
                    loss_enc = n_cut_loss(imgs, mask)
                    if step == "Train":
                        loss_enc.backward()
                        optimizer.step()
                    _enc_loss.append(loss_enc.item())

                    if step == "Train":
                        optimizer.zero_grad()  # zero the gradient buffers
                    recons = net.forward(imgs)  # return seg and attention map
                    loss_recons = recons_loss(imgs, recons)
                    if step == "Train":
                        loss_recons.backward()
                        optimizer.step()
                    _recons_loss.append(loss_recons.item())

            # epoch_enc_loss.append(np.array(_enc_loss).mean())
            # epoch_recons_loss.append(np.array(_recons_loss).mean())

            utils.print_gre("{}: \nEncoding loss: {:.3f}\t Reconstruction loss: {:.3f}".format(
                step, np.array(_enc_loss).mean(), np.array(_recons_loss).mean()
            ))
        scheduler.step()
        # save_model(net, np.array(_enc_loss).mean())


if __name__ == "__main__":
    args = utils.get_args()
    train(
        BASE_PATH + "patches_tries/",
        config=args,
        epochs=args.epochs,
    )