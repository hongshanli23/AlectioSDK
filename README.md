# Alectio SDK

AlectioSDK is a package that enables developers to build ML pipeline as a Flask app to interact with Alectio's
platform. 
It is designed for Alectio's clients, who perfer to keep their model and data on their on server. 

The package is under active development. More functionalities that aim to enhance robustness will be added soon
As for now the package provides a class `alectio_sdk.flask_wrapper.Pipeline` that inferfaces with customer-side
processes in a consistent manner. Customers need to implement 3 processes as python function:

* A process to train the model
* A process to test the model
* A process to apply the model to infer on unlabeled data

### train the model
The logic for training the model should be implemented in this process. The function should look like

```python
def train(payload):
    # get indices of the data to be trained
    labeled = payload['labeled']
    
    # get checkpoint to resume from
    resume_from = payload['resume_from']
    
    # get checkout to save for this loop
    ckpt_file = payload['ckpt_file']
    
    # implement your logic to train the model
    # with the selected data indexed by `labeled`
    return
    
```

The name of the function can be anything you like. It takes an argument `payload`, which is a 
dictionary with 3 keys

| key | value |
| --- | ----- |
| resume_from | a string that specifies which checkpoint to resume from |
| ckpt_file | a string that specifies the name of checkpoint to be saved for the current loop |
| labeled | a list of indices of selected samples used to train the model in this loop | 

Depending on your situation, the samples indicated in `labeled` might not be labeled (despite the variable
name). We call it `labaled` because in the active learning setting, this list represents the pool of 
samples iteratively labeled by the human oracle. 


### Test the model
The logic for testing the model should be implemented in this process. The function representing this 
process should look like

```python
def test(payload):
    # the checkpoint to test
    ckpt_file = payload['ckpt_file']
    
    # implement your testing logic here
    
    
    # put the prediction and label into 
    # two dictionaries
    
    # lbs <- dictionary of indices of test data and their ground-truth
    
    # prd <- dictionary of indices of test data and their prediction
    
    return {'predictions': prd, 'labels': lbs}
```
The test function takes an argument `payload`, which is a dictionary with 1 key

| key | value |
| --- | ----- | 
| ckpt_file | a string that specifies which checkpoint to test | 

The test function needs to return a dictionary with two keys

| key | value |
| --- | ----- | 
| predictions | dictionary of index of test sample and its prediction |
| labels | dictionary of index of test sample and its true label |

The format of the values depends on the type of ML problem. Please refer to the official
[examples](./examples) for details

## Apply inference
The logic for applying the model to infer on the unlabeled data should be implemented in this process. 
The function representing this process looks like
```python
def infer(payload):
    # get the indices of unlabeled data
    unlabeled = payload['unlabeled']
    
    # get the checkpoint file to be used for making inference
    ckpt_file = payload['ckpt_file']
    
    # implement your inference logic here
    
    
    # outputs <- save the output from the model on the unlabeled data as a dictionary
    return {'outputs': outputs}
```
The infer function takes an argument `payload`, which is a dictionary with 2 keys:
| key | value |
| --- | ----  | 
| ckpt_file | a string that specifies which checkpoint to use to infer on the unlabeled data | 
| unlabeled | a list of unlabeled data |

The `infer` function needs to return a dictionary with one key
| key | value |
| --- | ----- | 
| outputs | dictionary of index of unlabeled data and the model's output on it |
To take the most out of Alectio's platform, we suggest you return the 'rawest' output. For example, 
if it is a classification problem, return the output before applying softmax. 
For more details about the format of the output, please refer to the official [examples](./examples)


## Installation
```
git clone https://github.com/hsl89/AlectioSDK.git
cd AlectioSDK
pip install -r requirements.txt
python setup.py install
```

## Examples 
To help customers using this package, we provide detailed [examples](./examples) that covers a wide range of 
ML problems 





