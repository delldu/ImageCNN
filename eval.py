# coding=utf-8

"""
#
# /************************************************************************************
# ***
# ***	File Author: Dell, 2018年 09月 18日 星期二 16:09:52 CST
# ***
# ************************************************************************************/
#
"""

import os
import sys
import logging
import argparse

import model

parser = argparse.ArgumentParser(
    description='Evaluate Image Classificer Model')
parser.add_argument(
    '-root-dir',
    type=str,
    default=model.DEFAULT_VALID_DATA_ROOT_DIR,
    help='validating data root directory, default: ' +
    model.DEFAULT_VALID_DATA_ROOT_DIR)
parser.add_argument(
    '-batch-size',
    type=int,
    default=64,
    help='batch size for validating, default: 64')
parser.add_argument(
    '-model',
    type=str,
    default=model.DEFAULT_MODEL,
    help='trained model name, default: ' + model.DEFAULT_MODEL)
parser.add_argument(
    '-device',
    type=str,
    default="cuda:0",
    help='cuda:0 or cpu, default: cuda:0')

if __name__ == '__main__':
    args = parser.parse_args()

    if (not os.path.exists(args.root_dir)) or (not os.path.isdir(
            args.root_dir)):
        logging.error(args.root_dir + ' is not director or not exists.')
        sys.exit(-1)

    data = model.valid_data_loader(args.root_dir, args.batch_size)
    net = model.load_model(args.device, args.model)
    model.eval_model(args.device, net, data)
