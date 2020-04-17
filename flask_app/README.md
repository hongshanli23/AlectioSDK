# Tutorials for building customer-side ML pipeline
If you choose to leave your model and data on your own server, you will need to 
build a Flask app that enable us to trigger the training, testing and inference 
processes of your model. The file `app.py` we provide here handles all the networking
tasks of the Flask app. What left for you is to define functions that specify
you would want to train, test and apply inference on your model. In particular, you
will need to implement three functions `train(payload)`, `test(payload)` and `infer(payload)`.

All functions take `payload` as the argument. It is a python dictionary. For each function, 
I will explain what key-value pairs this dictionary entails. 

## train the model
The function `train(payload)` defines how you would want to train your model. 
The `payload` here contains three keys

| key | value |
| --- | ----- |
| resume_from | a string that specifies which checkpoint to resume from |
| ckpt_file | a string that specifies the name of checkpoint to be saved for the current loop |
| selected_indices | a list of indices of selected samples for this loop | 

Notice that `selected_indices` only includes the newly added samples. You will need to cache
the selected sample indices from previous loops on your end. 

For example, if the active learning process is at loop 3. Then you will need to resume from 
the checkpoint saved at loop 2. In this case, `resume_from = 'ckpt_2'` and `ckpt_file = 'ckpt_2'`. 
At the end of the training, you will need to save the checkpoint as `ckpt_3`. 


## test the model
The function `test(payload)` defines how you would want to test your model. 
The `payload` here contains one key
| key | value |
| --- | ----- | 
| ckpt_file | a string that specifies which checkpoint to test | 

The `test` function needs to return two dictionaries as a tuple
`(prd, lbs)` 

| return | about | 
| ------ | ----- | 
| prd | dictionary of indices of test samples and its prediction | 
| lbs | dictionary of indices of test samples and its true label | 

The `test` function needs to return 2 dictionaries `prd` and `lbs`.

### prd
It is a dictionary of test data indices and its prediction. 
The format of this dictionary depends on the type of ML problems.

#### Image classification
* key: indices of test data
* value: integer that representing the predicted class

#### Object detection
* key: indices of test data
* value: list of list. The inner list represents the predicted bounding
box. It should be of the format
```
[x1, y1, x2, y2, objectness, class]
```
| x1 | (int) x-coordinate of the top-left point of the bounding box |
| y1 | (int) y-coordinate of the top-left point of the bounding box | 
| x2 | (int) x-coordinate of the bottom-right point of the bounding box |
| y2 | (int) y-coordinate of the bottom-right point of the bounding box | 
| objectness | (float) objectness score ranging between 0 and 1 |
| class | (int) class label |

`x1, y1, x2, y2` need to be normalized by the image dimension. For example,

```
x1 = absolute value of the top-left point of the bbox / width of the image

y1 = aboslute value of the top-left point of the bbx / height of the image
```

### lbs
It is a dictionary of test data and its ground-truth label. 
The format of the values of
this dictionary is exactly the same as `prd`. 


## apply inference on the model
The last step in one active learning is to let the model infer on the 
unlabeled data. You will need to implement it through the `infer(payload)` 
function. 

The `payload` here has two keys
| key | value |
| --- | ----  | 
| ckpt_file | a string that specifies which checkpoint to use to infer on the unlabeled data | 
| unlabeled | a list of unlabeled data |

The `infer` function needs to return one dictionary `output`. 

### output
It is a dictionary of indices of unlabeled data and the model's inference.
The format of this dictionary depends on the type of ML problem

#### Image classification
* key: (int) indices unlabeled training set
* value: (list(float)) class probabilities 

#### Object detection 
* key: (int) indices of unlabeled training set
* value: (list(list)) prediction of each anchor box. The inner list should have the following
format
```
[x1, y1, x2, y2, objectness, c0, c2, ..., cn]
```
| x1 | (int) x-coordinate of the top-left point of the bounding box |
| y1 | (int) y-coordinate of the top-left point of the bounding box | 
| x2 | (int) x-coordinate of the bottom-right point of the bounding box |
| y2 | (int) y-coordinate of the bottom-right point of the bounding box | 
| objectness | (float) objectness score ranging between 0 and 1 |
| c0,...,cn | class distribution (n+1) classes in this case | 

Again, `x1, y1, x2, y2` needs to be normalized by the image dimension



