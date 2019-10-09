# Face anonymization in video

[Slovak](README.SK.md) version

The goal of this project is to detect, and blur faces in a video. Each face in a video is tracked and continuously anonymized. To detect face in a picture, we use both Haarcascade classifier and Tensorflow object detection API.

## Example of anonymized face
![Alt text](img/anonymFace.png?raw=true "Anonym face")

## Requirements
- Python 3
- OpenCV 4
- Tensorflow

## Body and faces models
In application, pretrained models for detection human body and face are used. Models are trained on COCO dataset and are part of Tensorflow library.

- faster_rcnn_inception_v2_coco 
  ( https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models )
- facessed_mobilenet_v2 (**models** directory)

## Script launch
Script is launched with next command. Also, path to video file source is required.
```
python faceAnonymizer.py -i <videoFile> -s
```

- -i <videoFile> - argument i is followed by path to video source file. This argument is required 
- -s – argument launches script in selfie mode. If only face near to a camera need to be anonymized, script can process video file faster.

## The process
Face anonymization in a video is processed in several steps. First two steps are initialization steps:
 
1. Script arguments parsing – to run script, it is needed to give an address of video to be processed. Also, script can run in selfie mode.
2. Load and initialization of Haarcascade classifiers and Tensorflow object detection API. In this step also, video file is loaded.
After initialization, steps below are done for each image in a video:
3. In this phase all bodies on an image are detected. After that, in the area of found bodies, faces are detected. If a face is found, it is anonymized. If not, upper part of a body is anonymized. Tensorflow object detection API is used to detect bodies and faces. If script was launched in selfie mode, whole this step is skipped, and processing is faster.
4. If any face is located near to a camera, body is not found but face have to be anonymized. Because of this, we detect face even in a part of image where no bodies are found. Tensorflow object detection is used in this step too.
5. In last step, Haarcascade classifiers are used to detect frontal faces, ears and eyes. Thank to these classifiers, we can detect profile face or a face which is half covered.
After these steps all people in a video are anonymized, no matter if their faces are near to camera or far. Even faces which are covered are blurred.
