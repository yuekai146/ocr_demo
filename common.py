#!/usr/bin/env mdl
import os
import hashlib
import getpass


class Config:

    minibatch_size = 1024
    fragment_size = 1
    nr_channel = 1
    image_shape = (1, 32, 100)
    height = 32
    width = 100
    hidden_dim = 4096

    num_pred = 4
    weight_decay = 1e-10
    lr_decay = 1e-6
    learning_rate = 0.1

    nr_epoch = 60

    num_test = 5000
    num_train = 100000
    noise_batch = 8

    blur_min=0.0
    blur_max = 1.0
    rotate_min = -10
    rotate_max = 10
    shear_min = -20
    shear_max = 20
    alpha_min = 0.0
    alpha_max = 50.0
    sigma = 5.0
    s_p_min = 0
    s_p_max = 0.15

    max_grad_norm = 1.0
    aug_h = 5
    aug_w = 5


config = Config()

# vim: ts=4 sw=4 sts=4 expandtab foldmethod=marker
