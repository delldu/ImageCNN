# coding=utf-8
#
# /************************************************************************************
# ***
# ***    File Author: Dell, 2018年 09月 18日 星期二 16:10:03 CST
# ***
# ************************************************************************************/
#

"""
data/train/lable1/...images.jpg
logs/project.model-epoch
"""

import os
import logging

import torch
import torchvision
import numpy as np

PROJECT = "flower"
DEFAULT_MODEL = "model/" + PROJECT + ".model"
DEFAULT_LABEL = "model/" + PROJECT + ".label"

DEFAULT_TRAIN_DATA_ROOT_DIR = "data/train"
DEFAULT_VALID_DATA_ROOT_DIR = "data/test"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class EpochLossAcc(object):
    def __init__(self, title):
        self.title = title
        self.vis = None
        self.win = None
        try:
            import visdom
            self.vis = visdom.Visdom(raise_exceptions=True)
        except:
            logging.info("Could not connect to visdom server, please make sure:")
            logging.info("1. install visdom:")
            logging.info("   pip insall visdom")
            logging.info("2. start visdom server: ")
            logging.info("   python -m visdom.server &")
            return
        self.win = self.vis.line(
            X=np.array([0]),
            Y=np.column_stack((np.array([0]), np.array([0]))),
            opts=dict(
                title=self.title + ' loss & acc',
                legend=['loss', 'acc'],
                width=1280,
                height=720,
                xlabel='Epoch',
            ))

    def plot(self, epoch, loss, acc):
        if self.vis is None or self.win is None:
            return

        self.vis.line(
            X=np.array([epoch]),
            Y=np.column_stack((np.array([loss]), np.array([acc]))),
            win=self.win,
            update='append')


def load_class_names(model_file=DEFAULT_MODEL):
    label_file = model_file.replace("model", "label")
    if not os.path.exists(label_file):
        label_file = DEFAULT_LABEL

    f = open(label_file)
    classnames = [line.strip() for line in f.readlines()]
    return classnames


def train_data_loader(datadir, batchsize):
    def save_class_names(classes):
        sep = "\n"
        f = open(DEFAULT_LABEL, 'w')
        f.write(sep.join(classes))
        f.close()

    T = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    ds = torchvision.datasets.ImageFolder(os.path.join(datadir), T)
    print("Training data information:")
    print(ds)
    print("Class names:", ds.classes)
    save_class_names(ds.classes)
    return torch.utils.data.DataLoader(ds, batch_size=batchsize, shuffle=True, num_workers=2)


def valid_data_loader(datadir, batchsize):
    T = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    ds = torchvision.datasets.ImageFolder(os.path.join(datadir), T)
    print("Evaluating data information:")
    print(ds)
    print("Class names:", ds.classes)
    return torch.utils.data.DataLoader(ds, batch_size=batchsize, shuffle=True, num_workers=2)


def image_to_tensor(image):
    T = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    t = T(image)
    t.unsqueeze_(0)
    return t


def load_squeezenet_model(device, name):
    classnames = load_class_names()

    if os.path.exists(name):
        model = torchvision.models.squeezenet1_1(pretrained=False)
    else:
        model = torchvision.models.squeezenet1_1(pretrained=True)

    c = model.classifier[1]
    c.out_channels = len(classnames)
    model.classifier[1] = c

    if os.path.exists(name):
        model.load_state_dict(torch.load(name))

    model = model.to(device)

    return model


def load_resnet18_model(device, name):
    classnames = load_class_names()

    if os.path.exists(name):
        model = torchvision.models.resnet18(pretrained=False)
    else:
        model = torchvision.models.resnet18(pretrained=True)

    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(classnames))

    if os.path.exists(name):
        model.load_state_dict(torch.load(name))

    model = model.to(device)

    return model

def load_model(device, name):
    return load_resnet18_model(device, name)
    #return load_squeezenet_model(device, name)


def train_model(device, model, dataloader, epochs):
    def save_model(model, epoch):
        name = "logs/{:s}.model-{:d}".format(PROJECT, epoch)
        logging.info('Saving model to ' + name + '...')
        torch.save(model.state_dict(), name)

    def save_steps(epochs):
        n = int((epochs + 1)/10)
        if n < 10:
            n = 10
        n = 10 * int((n + 9) / 10)  # round to 10x times
        return n

    logging.info("Start training ...")

    criterion = torch.nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 10 epochs
    dec_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.train()  # Set model to training mode

    viz = EpochLossAcc("Training")
    save_interval = save_steps(epochs)
    for epoch in range(epochs):
        dec_lr_scheduler.step()

        trainning_loss = 0.0
        correct, total = 0, 0

        # Iterate over data.
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # statics
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            trainning_loss += loss.item()

            trainning_acc = 100.0 * correct/total
        logging.info('Training epoch: %d/%d, loss: %12.4f, acc: %10.2f' %
                         (epoch + 1, epochs, trainning_loss, trainning_acc))

        viz.plot(epoch, trainning_loss, trainning_acc)

        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            save_model(model, epoch + 1)

    logging.info("Trainning finished.")


def eval_model(device, model, dataloader):
    logging.info("Start evaluating ...")

    training = model.training
    model.eval()  # Set model to evaluate mode

    correct, total = 0, 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            x, predicted = torch.max(outputs.data,
                                     1)  # by 0 -- cols, 1 -- rows
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train(mode=training)

    logging.info('Evaluating ACC: %10.2f%%' % (100.0 * correct / total))
    logging.info("Evaluating finished.")

    return correct / total


def model_predict(device, model, image):
    t = image_to_tensor(image)
    t = t.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(t)
        _, label = torch.max(outputs.data, 1)  # by 0 -- cols, 1 -- rows
        i = label[0].item()
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        prob = outputs[0][i].item()
    return i, prob
