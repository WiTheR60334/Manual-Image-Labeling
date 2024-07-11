# Gigazit Take-Home Challenge: Raw Image to Labeled Image Conversion

Image labeling in multiple annotation formats:
- PASCAL VOC (= [darkflow](https://github.com/thtrieu/darkflow))
- [YOLO darknet](https://github.com/pjreddie/darknet)


## Table of contents

- [Quick start](#quick-start)
- [Prerequisites](#prerequisites)
- [Run project](#run-project)
- [GUI usage](#gui-usage)
- [Authors](#authors)

## Quick start

To start using the YOLO Bounding Box Tool you need to clone the repo:

```
git clone --recurse-submodules git@github.com:WiTheR60334/Manual-Image-Labeling.git
```

### Prerequisites

You need to install:

- [Python](https://www.python.org/downloads/)
- [OpenCV](https://opencv.org/) version >= 3.0
    `pip install opencv-python opencv-contrib-python numpy tqdm lxml`

Alternatively, you can install everything at once by simply running:

```
pip install -U -r requirements.txt
```
- [PyTorch](https://pytorch.org/get-started/locally/) 
    Visit the link for a configurator for your setup.
    
### Run project

Step by step:

  1. Open the `main/` directory
  2. Insert the input images and videos in the folder **input/**
  3. Insert the classes in the file **class_list.txt** (one class name per line)
  4. Run the code.
  5. You can find the annotations in the folder **output/**

         python main.py [-h] [-i] [-o] [-t] [--tracker TRACKER_TYPE] [-n N_FRAMES]

         optional arguments:
          -h, --help                Show this help message and exit
          -i, --input               Path to images and videos input folder | Default: input/
          -o, --output              Path to output folder (if using the PASCAL VOC format it's important to set this path correctly) | Default: output/
          -t, --thickness           Bounding box and cross line thickness (int) | Default: -t 1
          --tracker tracker_type    tracker_type being used: ['CSRT', 'KCF','MOSSE', 'MIL', 'BOOSTING', 'MEDIANFLOW', 'TLD', 'GOTURN', 'DASIAMRPN']
          -n N_FRAMES               number of frames to track object for
  To use DASIAMRPN Tracker:
  1. Install the [DaSiamRPN](https://github.com/foolwood/DaSiamRPN) submodule and download the model (VOT) from [google drive](https://drive.google.com/drive/folders/1BtIkp5pB6aqePQGlMb2_Z7bfPy6XEj6H)
  2. copy it into 'DaSiamRPN/code/'
  3. set default tracker in main.py or run it with --tracker DASIAMRPN


#### How to use the deep learning feature
- I had used the tensorflow object detection code for auto labeling.
- Download one or some deep learning models from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
  and put it into `object_detection/models` directory (you need to create the `models` folder by yourself). The outline of `object_detection` looks like that:
  + `tf_object_detection.py`
  + `utils.py`
  + `models/ssdlite_mobilenet_v2_coco_2018_05_09`

Download the pre-trained model by clicking this link http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz and put it into `object_detection/models`. Create the `models` folder if necessary. Make sure to extract the model.

  **Note**: Default model used in `main_auto.py` is `ssdlite_mobilenet_v2_coco_2018_05_09`. We can
  set `graph_model_path` in file `main_auto.py` to change the pretrain model
- Using `main_auto.py` to automatically label data first

  TODO: explain how the user can 

### GUI usage

Keyboard, press: 

<img src="https://github.com/Cartucho/OpenLabeling/blob/master/keyboard_usage.jpg">

| Key | Description |
| --- | --- |
| a/d | previous/next image |
| s/w | previous/next class |
| e | edges |
| h | help |
| q | quit |

Video:

| Key | Description |
| --- | --- |
| p | predict the next frames' labels |

Mouse:
  - Use two separate left clicks to do each bounding box
  - **Right-click** -> **quick delete**!
  - Use the middle mouse to zoom in and out
  - Use double click to select a bounding box

## Authors

* **Romir Bedekar**