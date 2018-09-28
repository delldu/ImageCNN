ImageCNN
====

![](images/transform.jpg)

Dataset
----
1. **Download & unzip pictures under data directory**


```
wget http://download.tensorflow.org/example_images/flower_photos.tgz
```
**Examples**

| ![](images/daisy_5794835_d15905c7c8_n.jpg) | ![](images/dandelion_144040769_c5b805f868.jpg) | ![](images/roses_568715474_bdb64ccc32.jpg ) | ![](images/sunflowers_40410814_fba3837226_n.jpg) | ![](images/tulips_4838669164_ffb6f67139.jpg) |
| ------------------------------------------ | ---------------------------------------------- | ------------------------------------------- | ------------------------------------------------ | -------------------------------------------- |
| daisy                                          | dandelion                                              |roses                                           |sunflowers                                                |tulips                                           |

2. **Split files into two folders: train and test with tools/shuflink**
`cd data`
`../tools/shuflink flower_photos train test`
`cd ..`



Train
----

**`$python train.py -h`**

Training models are saved under logs.  After training, you should move one of them to model directory and rename it as “flower.model“ for evaluation and prediction.



Evaluate
----

**`$ python eval.py -h`**

**`$ python eval.py`** 
`Evaluating data information:`
`Dataset ImageFolder`
    `Number of datapoints: 918`
    `Root Location: data/test`
    `Transforms (if any): Compose(`
                             `Resize(size=256, interpolation=PIL.Image.BILINEAR)`
                             `CenterCrop(size=(224, 224))`
                             `ToTensor()`
                             `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
                         `)`
    `Target Transforms (if any): None`
`Class names: ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']`
`2018-09-28 15:31:19,234 - model.py[line:203] - INFO: Start evaluating ...`
`2018-09-28 15:31:21,553 - model.py[line:222] - INFO: Evaluating ACC:  96.30%`
`2018-09-28 15:31:21,553 - model.py[line:223] - INFO: Evaluating finished.`

Predict
----

**`python predict.py -h`**

`python predict.py images/daisy_5794835_d15905c7c8_n.jpg` 

`Image class: 0, daisy, 1.00, images/daisy_5794835_d15905c7c8_n.jpg`

![](images/daisy_5794835_d15905c7c8_n.jpg)

Requirements
----

- Python 3

- Pytorch 0.4.0

- Torchvision 0.2.1

- Pillow 4.2.1



License
----

ImageCNN is released under the [Apache 2.0 license]



Chinese Document
----
**通用图像分类器.**

