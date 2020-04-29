'''
Main processes for the projecct

Author: Hongshan Li
Email: hongshan.li@alectio.com
'''

import torch
import torch.optim as optim
import torch.nn as nn
from torchtext import data
from torch.utils.tensorboard import SummaryWriter

import time
import os
from collections import Counter
import pickle

from dataset import TEXT, LABEL, DailyDialog, entire_data
from model import RNN
import envs


# initialze the model
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 10
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(INPUT_DIM, 
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT, 
            PAD_IDX)

model.embedding.weight.data.copy_(TEXT.vocab.vectors)

def accuracy(outputs, targets):
    '''compute accuracy of a batch prediction'''
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == targets).float().sum()
    return correct / outputs.shape[0]
    

def train(payload):
    '''Train the model and save the checkpoint
    
    payload: the payload object received from the http call
        from the Alectio Platform. 
        It is parsed as an immutable dictionary with 3 keys
        
        labeled: list
            indices of training data to be used for this active learning loop
        
        resume_from: str
            checkpoint file to resume from. For example, in loop n 
            of active learning, the value of this key is `ckpt_(n-1)`, 
            indicating that you should resume from ckeckpoint saved in loop n-1
        
        ckpt_file: str
            ckeckpoint file to save. For example, in loop n of active 
            learing, the value of this key is `ckpt_n`, i.e. you 
            should save the model ckeckpoint as `ckpt_n` in your log directory
    
    '''
    
    # which checkpoint to resume from
    resume_from = payload['resume_from']
    
    # which checkpoint to save as
    ckpt_file = payload['ckpt_file']
    
    # indices of data to train in this loop
    labeled = payload['labeled']
    
    # training hyperparameters:
    batch_size=128 # batch size
    lr=1e-2 # learning rate
    weight_decay=1e-4 # weight decay
    epochs = 2 # number of epochs. Use a more realistic number in production
    print_fq = 20 # print progress per 20 steps
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    
    # resume model and optimizer from ckpt of the previous loop
    if resume_from is not None:
        ckpt = torch.load(os.path.join(envs.EXPT_DIR, resume_from))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    
    # loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # build a dataset from data indexed by labeled
    labeled = [int(ix) for ix in labeled]
    
    dstr = DailyDialog(
        path=os.path.join(envs.DATA_DIR, 'train.json'),
        text_field=TEXT,
        label_field=LABEL,
        samples=labeled)
   
    
    # build a data loader
    ldtr = data.BucketIterator(
        dstr,
        batch_size=batch_size,
        device=envs.DEVICE,
        shuffle=True,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True)
    
    # set to train mode
    model.train()
    
    # log the epoch performance for further analysis
    writer = SummaryWriter(envs.EXPT_DIR)
    
    steps = 0
    for epoch in range(epochs):
        for batch in ldtr:
            text, text_length = batch.text
            outputs = model(text, text_length)
            
            loss = loss_fn(outputs, batch.label.long())
            acc = accuracy(outputs, batch.label.long())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # log to the tensorboard
            writer.add_scalar('Loss/train', loss.item(), steps)
            writer.add_scalar('Loss/accuracy', acc.item(), steps)
            
            steps+=1
    
            if steps % print_fq == 0:
                print("Epoch: {}, Global Step: {}, Loss: {}".format(epoch, step, loss.item()))
          
        # save ckpt every epoch
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(ckpt, os.path.join(envs.EXPT_DIR, ckpt_file))

    writer.close()                 
    return

def test(payload):
    '''Test the model and return the predictions and ground-truth
    
    payload: the payload object received from the http call
        from the Alectio Platform. 
        It is parsed as an immutable dictionary with 1 key
        
        ckpt_file: str
            The model ckeckpoint file to be tested. For example,
            in loop n of active learning, the value of this key is 
            `ckpt_n`, indicating that you should load your model
            from `ckpt_n` in the log directory and test it
    ''' 
    
    # which ckpt to test                  
    ckpt_file = payload['ckpt_file']

    # create test set
    dsts = DailyDialog(
        path=os.path.join(envs.DATA_DIR, 'test.json'),
        text_field=TEXT,
        label_field=LABEL,
        samples=None)
    
    batch_size=256
    
    # create a data loader
    # use a regular data loader without sorting samples according to its length
    # this is because we need to build a dictionary of data index and its prediction
    # so we cannot disrupt the order
    ldts = data.Iterator(
        dsts,
        batch_size=batch_size,
        device=envs.DEVICE,
        shuffle=False)
   
    # ground-truth labels and predictions
    lbs, prd = [], []
    
    model.eval()
    with torch.no_grad():
        for batch in ldts:
            text, text_length = batch.text
            outputs = model(text, text_length)
            preds = torch.argmax(outputs, dim=1)
            
            lbs.extend(batch.label.long().cpu().numpy().tolist())
            prd.extend(preds.cpu().numpy().tolist())
            
    # convert lbs and prd to dict
    lbs = {i:v for i, v in enumerate(lbs)}
    prd = {i:v for i, v in enumerate(prd)}
    
    return {"predictions": prd, "labels": lbs}


def infer(payload):
    '''Use the model to infer on the unlabeled data and return the output
    
    payload: the payload object received from the http call
        from the Alectio Platform. 
        It is parsed as an immutable dictionary with 2 keys
        
        ckpt_file: str
            The checkpoint file to use to apply inference. 
            For example, in loop n of active learning, the
            value of this key is `ckpt_n`. It means you should
            load `ckpt_n` from the log directory to your model
            for inference.
            
        unlabeled: list
            indices of the data in the training set to be used
            for inference
    '''
    ckpt_file = payload['ckpt_file']
    unlabeled = payload['unlabeled'] 
    
    # which ckpt to use to infer              
    ckpt_file = payload['ckpt_file']
    
    # load model state dict
    ckpt = torch.load(os.path.join(envs.EXPT_DIR, ckpt_file))
    model.load_state_dict(ckpt['model'])
    

    # create dataset
    dsinf = DailyDialog(
        path=os.path.join(envs.DATA_DIR, 'train.json'),
        text_field=TEXT,
        label_field=LABEL,
        samples=unlabeled)
    
    batch_size=256
    
    # create a data loader
    # use a regular data loader without sorting samples according to its length
    # this is because we need to build a dictionary of data index and its output
    # so we cannot disrupt the order
    ldinf = data.Iterator(
        dsinf,
        batch_size=batch_size,
        device=envs.DEVICE,
        shuffle=False)
    
    outputs = None
    model.eval()
    with torch.no_grad():
        for batch in ldinf:
            text, text_length = batch.text
            _outputs = model(text, text_length)
            
            if outputs is None:
                outputs = _outputs
            else:
                outputs = torch.cat([outputs, _outputs], dim=0)
    
    # convert outputs to list then to dict to be send to Alectio server
    outputs = outputs.cpu().numpy().tolist()
    outputs = {ix:d for ix, d in zip(unlabeled, outputs)}
    return {"outputs": outputs}

if __name__ == '__main__':
    # debug
    payload = {
        'labeled': list(range(10)),
        'ckpt_file': 'ckpt_0',
        'resume_from': None,
    }
    print('Training')
    train(payload)
    
    payload = {
        'ckpt_file': 'ckpt_0',
    }
    print('Testing')
    test(payload)
        
    print('Infering')
    payload = {
        'ckpt_file': 'ckpt_0',
        'unlabeled': list(range(10))
    }
    infer(payload)
    
                                   
                                   
                              
                                   
