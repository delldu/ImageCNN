# coding=utf-8
#
# /************************************************************************************
# ***
# ***	File Author: Dell, 2018年 09月 18日 星期二 16:09:58 CST
# ***
# ************************************************************************************/
#

import argparse
from PIL import Image
import model

parser = argparse.ArgumentParser(description='Image Classifier Predictor')
parser.add_argument(
    '-model',
    type=str,
    default=model.DEFAULT_MODEL,
    help='trained model [' + model.DEFAULT_MODEL + ']')
parser.add_argument(
    '-device',
    type=str,
    default="cuda:0",
    help='cuda:0 or cpu [cuda:0]')
parser.add_argument('images', type=str, nargs='+', help='image files')

if __name__ == '__main__':
    args = parser.parse_args()

    net = model.load_model(args.device, args.model)
    classnames = model.load_class_names(args.model)

    for i in range(len(args.images)):
        image = Image.open(args.images[i]).convert('RGB')
        label, prob = model.model_predict(args.device, net, image)
        print('Image class: %d, %s, %.2f, %s' % (label, classnames[label],
                                                 prob, args.images[i]))
