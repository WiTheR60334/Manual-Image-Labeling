# Gigazit Take-Home Challenge: Raw Image to Labeled Image Conversion



- I've developed two scripts for image labeling: a manual labeling tool, inspired by YOLO that allows users to define and assign classes to images manually. Additionally, I've created an automation labeling script , which utilizes TensorFlow's object detection capabilities to automatically label images based on detected objects.

## Project Overview

- My project integrates advanced object detection and tracking capabilities using deep learning models and the `DaSiamRPN algorithm`. The main components are divided across several files: `main.py` manages the core functionality, coordinating image input and output through the input/ and output/ directories. Configuration settings in config.ini define object detection parameters, including score thresholds and specific object IDs. The `dasiamrpn.py` file encapsulates the DaSiamRPN tracker, enabling real-time object tracking across frames.

## Manual Labeling Process

- The project features a robust manual labeling system inspired by `YOLO`(You Only Look Once) methodology. Users annotate images by placing bounding boxes around objects of interest, enhancing the dataset for training models.

## Addtional Features

- An additional automation labeling script leverages TensorFlow's object detection capabilities, providing automated annotation to streamline data preparation. This dual approach ensures comprehensive labeling, crucial for training accurate object detection models in applications such as surveillance, robotics, and automated systems.

Image labeling in multiple annotation formats:
- PASCAL VOC (= [darkflow](https://github.com/thtrieu/darkflow))
- [YOLO darknet](https://github.com/pjreddie/darknet)


## Table of contents

- [Quick start](#quick-start)
- [Prerequisites](#prerequisites)
- [Run project](#run-project)
- [GUI usage](#gui-usage)
- [Working](#working)
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

## Working

- If you paste your own image in input dir for labeling , then you need to define the classes in each new line like this : 

![image](https://github.com/WiTheR60334/Manual-Image-Labeling/assets/115364885/83b1840b-ae49-4fe0-a5cc-3ffcf20aa084)

- I would be using these 3 classes as example.

- From below image we can clearly see that we can manually rectangle the person and label it.
  
![Screenshot 2024-07-11 164616](https://github.com/WiTheR60334/Manual-Image-Labeling/assets/115364885/90fcf18b-4ee4-4461-8f37-a7209267967e)


- To change the class, press `w` and manually rectangle the object.
  
![Screenshot 2024-07-11 164705](https://github.com/WiTheR60334/Manual-Image-Labeling/assets/115364885/54db8074-e4cc-4b73-b726-5e006f762e2f)

- We need to manually label each and every object related to that class for better performace of model.
  
![Screenshot 2024-07-11 164830](https://github.com/WiTheR60334/Manual-Image-Labeling/assets/115364885/6f51fd2a-290a-4167-a366-fae923c2b3af)

- You can clealry see that in img_1 of person, we had rectangled 5 people and those 5 peoples markings are saved in a txt file of output dir. Similarly goes for img 2 and 3 also.
  
![Screenshot 2024-07-11 164930](https://github.com/WiTheR60334/Manual-Image-Labeling/assets/115364885/e520cd4a-0411-43a2-a69b-d8c16ee81677)


- Labeled images are also saved in .xml format in output dir.

## Authors

* **Romir Bedekar**
