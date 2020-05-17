# Object detection on COCO

This example shows you how to build `train`, `test` and `infer` processes
for object detection problems. In particular, I will show you the format
of the return of `test` and `infer`. For object detection problem, those
return can be a little bit involved. But to get most out of Alectio's platform,
those returns needs to be correct. 

Since the goal of this tutorial is to show you how to setup a pipeline, 
we are not going to train on the entire COCO dataset. Instead, we will 
only use 100 samples for training set and 100 samples for test set. 
And we will use yolo-v3 as the model.

### 1. Make sure you are in the right environments
All the examples we provide here requires some python dependencies.
Follow the readme in the [examples directory](../)


### 2. Download pre-processed data from Alectio public resource S3 bucket
This step requires aws command line interface, which is part of python 
dependencies of this enviornment. 
Don't worry if you don't have an aws account, you won't need it to download
stuffs from public buckets

```
cd <this directory>
mkdir data
cd data
aws s3 cp s3://alectio-resources/cocosamples . --recursive
```

For most part the data is preprocessed according to the Darknet convention. The only difference
is that we use xyxy for ground-truth bounding box. 

### 3. Build train process
We will train a [Darknet yolov3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) for
this demo. The model is defined in `model.py` the configuration file that specifies the
architecture of the model is defined in `yolov3.cfg`

Refer to [AlectioSDK ReadMe](../../README.md) for general information regarding the 
arguments of this process.

The function is well-annotated, please go through the code for more detail. 

The return of `train` function is a dictionary
```
{"labels": lbs}
```

'lbs' os a dictionary whose keys are the indices of the labeled images. 
The value `lbs[i]` is a dictionary that records the ground-truth bounding 
boxes and class labels on the image `i`
**Keys of `lbs[i]`**

boxes
> *(list of list)* A list of ground-truth bounding boxes on image `i`.
    Each bounding box should normalized acoording to the dimension
    of test image `i` and it should be in xyxy-format
 
objects:
> *(list of int)* A list of class label of each ground-truth bounding box

We will use the labels on `labeled` data to provide insights on 
preferential learning

### 4. Build test process
The test process tests the model trained in each active learning loop.
In this example, the test process is the `test` function defined 
in `processes.py`. 

Refer to [AlectioSDK ReadMe](../../README.md) for general information regarding the 
arguments of this process.

#### Return of the test process 
You will need to run NMS on the predictions on the test images and return 
the final detections along with the ground-truth bounding boxes and objects
on each image. 

The return of `test` function is a dictionary 
```
    {"predictions": prd, "labels": lbs}
    
```

`prd` is a dictionary whose keys are the indices of the test 
images. The value of `prd[i]` is a dictionary that records the final
detections on test image `i`

**Keys of `prd[i]`**

boxes
> *(list of list)* A list of detected bouding boxes. 
    Each bounding box should be normalized according 
    to the dimension of test image `i` and it 
    should be in xyxy-format
  
scores:
> *(list of float)* A list of objectedness of each detected
   bounding box. Objectness should be in \[0, 1\]

objects:
> *(list of int)* A list of class label of each detected 
    bounding box. 


'lbs' os a dictionary whose keys are the indices of the test images. 
The value `lbs[i]` is a dictionary that records the ground-truth bounding 
boxes and class labels on the image `i`

**Keys of `lbs[i]`**

boxes
> *(list of list)* A list of ground-truth bounding boxes on image `i`.
    Each bounding box should normalized acoording to the dimension
    of test image `i` and it should be in xyxy-format
 
objects:
> *(list of int)* A list of class label of each ground-truth bounding box


  

### 5. Build infer process
Infer process is used to apply model on the unlabeled set to make inference. 
We will use the infered output to estimate which of those unlabeled data will
be most valuable to your model.

Refer to [AlectioSDK ReadMe](../../README.md) for general information regarding the 
arguments of this process.

#### Return of the infer process
The return of the infer process is a dictionary
```python
{"outputs": outputs}
```

`outputs` is a dictionary whose keys are the indices of the unlabeled
images. The value of `outputs[i]` is a dictionary that records the output of
the model on training image `i`. 

**Keys of `outputs[i]`**
boxes
> *(list of list)* A list of detected bouding boxes.
    You need to apply NMS on all predicted bounding 
    boxes. 
    Each bounding box should be normalized according 
    to the dimension of test image `i` and it 
    should be in xyxy-format
  
scores:
> *(list of float)* A list of objectedness of each detected
   bounding box. Objectness should be in \[0, 1\]

pre_softmax:
> *(list of list)* A list of class distribution for each 
    detected bounding boxes. As the name suggests, do not
    apply softmax



