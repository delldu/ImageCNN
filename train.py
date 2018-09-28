# coding=utf-8
#
# /**************************************************************************
# ***
# ***	File Author: Dell, 2018年 09月 18日 星期二 16:28:12 CST
# ***
# **************************************************************************/
#

import os
import sys
import logging
import argparse

import model

parser = argparse.ArgumentParser(description='Train Image Classificer Model')

parser.add_argument(
    '-root-dir',
    type=str,
    default=model.DEFAULT_TRAIN_DATA_ROOT_DIR,
    help='train data root directory, default: ' +
    model.DEFAULT_TRAIN_DATA_ROOT_DIR)
parser.add_argument(
    '-epochs',
    type=int,
    default=32,
    help='number of epochs for train, default: 32')
parser.add_argument(
    '-batch-size',
    type=int,
    default=64,
    help='batch size for training, default: 64')
parser.add_argument(
    '-device',
    type=str,
    default="cuda:0",
    help='cuda:0 or cpu, default: cuda:0')


def makedirs():
    for d in ["logs", "model"]:
        if not os.path.exists(d):
            os.mkdir(d)
        if not os.path.isdir(d):
            logging.error(
                "Please create dir 'logs' or 'model' under current directory.")
            raise Exception("logs or model is not directory.")


if __name__ == '__main__':
    args = parser.parse_args()

    if (not os.path.exists(args.root_dir)) or (not os.path.isdir(
            args.root_dir)):
        logging.error(args.root_dir + ' is not director or not exists.')
        sys.exit(-1)

    makedirs()

    data = model.train_data_loader(args.root_dir, args.batch_size)
    net = model.load_model(args.device, model.DEFAULT_MODEL)
    model.train_model(args.device, net, data, args.epochs)
