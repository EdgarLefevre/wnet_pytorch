# -*- coding: utf-8 -*-
import os

import numpy as np
import progressbar
import skimage.io as io
import sklearn.model_selection as sk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from wnet.models import wnet, residual_wnet
from wnet.utils import utils, data, soft_n_cut_loss, ssim

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

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

if os.uname()[1] == "iss":
    BASE_PATH = "/home/edgar/Documents/Datasets/JB/new_images/"
else:
    BASE_PATH = "/home/elefevre/Datasets/JB/new_images/"

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


def _step(net, step, dataset, optim, recons_loss, n_cut_loss, epoch, config):
    _enc_loss, _recons_loss = [], []
    if step == "Train":
        net.train()
    else:
        net.eval()
    with progressbar.ProgressBar(max_value=len(dataset), widgets=widgets) as bar:
        for i in range(len(dataset)):  # boucle inf si on ne fait pas comme ça
            bar.update(i)
            imgs = dataset[i].cuda()
            if step == "Train":
                optim.zero_grad()
            recons, mask = net.forward(imgs)
            loss_recons = recons_loss(imgs, recons)
            loss_enc = n_cut_loss(imgs, mask)
            if step == "Train":
                loss = loss_enc + loss_recons
                loss.backward()
                optim.step()
            _enc_loss.append(loss_enc.item())
            _recons_loss.append(loss_recons.item())
            if step == "Validation" and (epoch + 1) == config.epochs:
                utils.visualize(net, imgs, epoch + 1, i, config, path="data/results/")
    return _enc_loss, _recons_loss


# def train_att(path_imgs, config, epochs=5):  # todo: refactor this ugly code
#     net = wnet.Wnet_attention(filters=config.filters, drop_r=config.drop_r).cuda()
#     optimizer = optim.Adam(net.parameters(), lr=config.lr)
#     n_cut_loss = soft_n_cut_loss.NCutLoss2D()
#     recons_loss = nn.MSELoss()
#     # recons_loss = ssim.ssim
#     #  get dataset
#     dataset_train, dataset_val = get_datasets(path_imgs, config)
#     epoch_enc_train = []
#     epoch_recons_train = []
#     epoch_enc_val = []
#     epoch_recons_val = []
#
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, epochs, verbose=True
#     )
#     for epoch in range(epochs):
#         _enc_loss = []
#         _recons_loss = []
#         utils.print_gre("Epoch {}/{}".format(epoch + 1, epochs))
#         for step in ["Train", "Validation"]:
#             if step == "Train":
#                 net.train()
#                 dataset = dataset_train
#             else:
#                 net.eval()
#                 dataset = dataset_val
#             with progressbar.ProgressBar(max_value=len(dataset), widgets=widgets) as bar:
#                 for i in range(len(dataset)):  # boucle inf si on ne fait pas comme ça
#                     bar.update(i)
#                     imgs = dataset[i].cuda()
#                     if step == "Train":
#                         optimizer.zero_grad()  # zero the gradient buffers
#                     mask, att = net.forward_enc(imgs)  # return reconstruction
#                     recons = net.forward(imgs)  # return seg and attention map
#                     loss_enc = n_cut_loss(imgs, mask)
#                     loss_recons = recons_loss(imgs, recons)
#                     if step == "Train":
#                         total_loss = loss_enc + loss_recons
#                         total_loss.backward()
#                         optimizer.step()
#                     _enc_loss.append(loss_enc.item())
#                     _recons_loss.append(loss_recons.item())
#             if step == "Train":
#                 epoch_enc_train.append(np.array(_enc_loss).mean())
#                 epoch_recons_train.append(np.array(_recons_loss).mean())
#             else:
#                 epoch_enc_val.append(np.array(_enc_loss).mean())
#                 epoch_recons_val.append(np.array(_recons_loss).mean())
#
#             utils.print_gre("{}: \nEncoding loss: {:.3f}\t Reconstruction loss: {:.3f}".format(
#                 step, np.array(_enc_loss).mean(), np.array(_recons_loss).mean()
#             ))
#         scheduler.step()
#         utils.visualize_att(net, imgs, epoch+1, config)
#     utils.learning_curves(epoch_enc_train, epoch_recons_train, epoch_enc_val, epoch_recons_val)
#     # save_model(net, np.array(_enc_loss).mean())

def reconstruction_loss(imgs, recons):
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    # ssim_loss = ssim.ssim
    return mse(recons, imgs) #+ bce(recons, imgs)


def train(path_imgs, config, epochs=5):  # todo: refactor this ugly code
    # net = wnet.WnetSep(filters=config.filters, drop_r=config.drop_r).cuda()
    net = residual_wnet.Wnet_Seppreact(filters=config.filters, drop_r=config.drop_r).cuda()
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    n_cut_loss = soft_n_cut_loss.NCutLoss2D()
    recons_loss = reconstruction_loss
    #  get dataset
    dataset_train, dataset_val = get_datasets(path_imgs, config)
    epoch_enc_train = []
    epoch_recons_train = []
    epoch_enc_val = []
    epoch_recons_val = []

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, verbose=True
    )
    for epoch in range(epochs):
        _enc_loss = []
        _recons_loss = []
        utils.print_gre("Epoch {}/{}".format(epoch + 1, epochs))
        for step in ["Train", "Validation"]:
            if step == "Train":
                dataset = dataset_train
            else:
                dataset = dataset_val
            _enc_loss, _recons_loss = _step(net, step, dataset, optimizer, recons_loss, n_cut_loss, epoch, config)
            if step == "Train":
                epoch_enc_train.append(np.array(_enc_loss).mean())
                epoch_recons_train.append(np.array(_recons_loss).mean())
            else:
                epoch_enc_val.append(np.array(_enc_loss).mean())
                epoch_recons_val.append(np.array(_recons_loss).mean())

            utils.print_gre("{}: \nEncoding loss: {:.3f}\t Reconstruction loss: {:.3f}".format(
                step, np.array(_enc_loss).mean(), np.array(_recons_loss).mean()
            ))
        scheduler.step()
    utils.learning_curves(epoch_enc_train, epoch_recons_train, epoch_enc_val, epoch_recons_val)


if __name__ == "__main__":
    args = utils.get_args()
    train(
        BASE_PATH + "patches_tries/",
        config=args,
        epochs=args.epochs,
    )
