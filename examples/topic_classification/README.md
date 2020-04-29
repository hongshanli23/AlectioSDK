# Topic Classification on DailyDialog Dataset

This example is intended to show you how to build `train`, `test` and `infer` processes for `AlectioSDK` for topic
classification problems. We will use [DailyDialog](https://arxiv.org/abs/1710.03957) dataset. Each sample in this
dataset is a conversation between two persons. The objective is to classify the topic of their converstaion. The topics are labeled as following:

| label | topic |
| ----- | ----- |
| 0    | Ordinary Life | 
| 1     | School Life | 
| 2    | Culture & Education | 
| 3    | Attitude & Emotion | 
| 4    | Relationship |
| 5     | Tourism | 
| 6    | Health | 
| 7    | Work |
| 8     | Politics | 
| 9     | Finance | 

Since the size of this dataset is not too big, we have preprocessed in and 
put it in `./data` direcotory. 


### Install dependencies 
We will create a virtual environment for this project and install dependencies
into the virtual environment. 

1. Go to the root directory of this project and install `virtualenv`
```
cd <AlectioSDK root>/examples/topic_classification
pip install virtualenv
```

2. Create a virtual environment and install dependencies
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

We use `spacy` to parse the text data. The module was already
installed in this, but we still need to download the English 
model for it
```
python -m spacy download en
```

3. Install `AlectioSDK` to this environment
```
cd <AlectioSDK root>
python setup.py install
```

4. Download GloVe vectors
We will use GloVe 6B.100d for word embedding. 
Create a direcotory in this project to save 
the vectors
```
mkdir vector
```
Then download the GloVe vectors and unzip it
```
cd vector
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```
After you unzipped it, the directory structure of 
`vector` should look like
```
|-- vector 
|   |-- glove.6B.50d.txt
|   |-- glove.6B.100d.txt
|   |-- glove.6B.200d.txt
|   |-- glove.6B.300d.txt
```
Since we are only going to use `glove.6B.100.txt`, you can delete the 
rest if you prefer to. 

5. Create a log direcotory
Create a log directory in the project root to save checkpoints
```
cd <AlectioSDK root>/examples/topic_classification
mkdir log
```

6. Source the envrionment variables into the current shell session. 
We will need some environment variables for this project. They are located
at `./setenv/sh`. Let's take a closer look at those variable

| variable | meaning | 
| -------- | ------- |
| VECTOR_DIR | directory where word vectors are saved |
| EXPT_DIR | directory where experiment log and checkpoints are saved |
| DATA_DIR | direcotry where processed data is saved | 
| DEVICE   | cpu or gpu device where model training/testing/inference takes place | 
| FLASK_ENV | the environment for the Flask app we build |

The default setting of those variables should work. Source those variables
in the shell
```
source setenv.sh
```

7. Build dataset object
We will use `pytorch` and `torchtext` for this project. We build a Dataset
object `DailyDialog`, text and label fields there. Please checkout the code
in `./dataset.py` for more detail

8. Build a model
We will use a 2-layer bidirectional LSTM for text classfication. For
the architecture of the model see `./model.py`


9. Build train/test/infer processes
Now, we are in the most important stage of the project, building 
`train/test/infer` processes for the project.
Please checkout the `train`, `test`, `infer` functions in `./processes.py`
Each block of logic in those functions are well commented and you should be
able to understand what those processes entail. 


You can try to see those processes in action by 
```python
python processes.py
```

10. Wrap up the processes into a Flask app
The final touch of the project is in `main.py`, where we wrap up processes
defined above into `alectio_sdk.flask_wrapper.Pipeline`

You can start the app by 
```python
python main.py
```

### Format of the `test/infer` process return
The return from the `test` and `infer` will be sent to Alectio's platform for 
computing the model performance and making active learning decisions. 
Therefore, it is important to make sure the format of those returns are correct


#### Return from `test` process
Return from the `test` process will be used to compute perforamnce metrics of
your model. The return is a dictionary with two keys

| key | value |
| --- | ----- | 
| predictions | a dictionary of test data index and model's prediction |
| labels | a dictionary of test data index and its ground-truth label | 

For classficatio problem like this project, the model's prediction and 
the ground-truth of each test sample is an integer indicating the class label.

For example, if the test set consists of $n+1$ samples indexed by $0, 1, \cdots, n$.
Then the value of `predictions` looks like
```python
{
    0: x0,
    1: x1,
    ...
    n: xn
}
```
where $xi$ is the integer class label of sample $i$ predicted by the model. 

The value of `labels` looks like
```python
{
    0: y0,
    1: y1,
    ...
    n: yn
}
```
where $yi$ is the integer ground-truth class label of sample $i$






