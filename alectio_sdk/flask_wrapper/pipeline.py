from flask import jsonify
from flask import Flask, Response
from flask import request
from flask import send_file

import traceback
import sys
import os
import time
import json
from copy import deepcopy

from .s3_client import S3Client

class Pipeline(object):
    def __init__(self, name, train_fn, test_fn, infer_fn):
        self.app = Flask(name)

        self.train_fn = train_fn
        self.test_fn = test_fn
        self.infer_fn = infer_fn
    
        self.client = S3Client()
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        
        with open(os.path.join(dir_path, 'config.json'), 'r') as f:
            self.config = json.load(f)
            
        # one loop
        self.app.add_url_rule(
            '/one_loop', 'one_loop', self.one_loop, methods=['POST']
            )
        
    
    def one_loop(self):
        payload = request.json()
        self._one_loop(payload)
        
    
    def _one_loop(self, payload):
        ''' Execute one loop of active learning '''
        
        # get some global args here
        self.logdir = payload['experiment_id']
        if not os.path.isdir(self.logdir):
            os.mkdir(self.logdir)
    
        # read selected indices upto this loop
        self.cur_loop = payload['cur_loop'] 
        self.bucket_name=payload['bucket_name']
        
        # type of the ML problem
        self.type = payload['type']
        
        # dir for expt log in S3
        expt_dir = [payload['user_id'], payload['project_id'], 
                    payload['experiment_id']]
        
        if self.bucket_name==self.config['sandbox_bucket']:
            # shared S3 bucket for sandbox user
            self.expt_dir = os.path.join(payload['user_id'],
                                        payload['project_id'],
                                        payload['experiment_id'])
            
            self.project_dir = os.path.join(payload['user_id'],
                                           payload['project_id'])
        else:
            # dedicated S3 bucket for paid user
            self.expt_dir = os.path.join(payload['project_id'],
                                        payload['experiment_id'])
            
            self.project_dir = os.path.join(payload['project_id'])

        
        # get meta-data of the data set
        key = os.path.join(self.project_dir, "meta.json")
        self.meta_data = self.client.read(self.bucket_name, key, "json")
        
        
        if self.cur_loop == 0:
            self.resume_from= None 
        else:
            self.resume_from = 'ckpt_{}'.format(self.cur_loop-1)
      
        self.ckpt_file = 'ckpt_{}'.format(self.cur_loop)
        
        self.train()
        self.test()
        self.infer()
        
        
    
    def train(self):
        ''' run customer training process 
        fetch selected indices upto current loop 
        use expt_id to create logdir
        pass three keyward arguments to the customer defined logic
        
        labeled: labeled indices so far
        resume_from: 
        ckpt_file
        logdir
        
        if train_fn is executed successfully, write labels to S3,
            call PBE with success
        if train_fn failed, call PBE with error msg
        '''
        
        start = time.time()
        
        self.labeled = []
        for i in range(self.cur_loop+1):
            object_key = os.path.join(self.expt_dir, 
                                      'selected_indices_{}.pkl'.format(i))  
            
            selected_indices = self.client.read(self.bucket_name, 
                                        object_key=object_key, 
                                         file_format='pickle')
            self.labeled.extend(selected_indices)
        
        labels = self.train_fn(labeled=deepcopy(self.labeled), 
                               resume_from=self.resume_from, 
                               ckpt_file=self.ckpt_file,
                               logdir=self.logdir)
        
        end = time.time()
        
        # @TODO compute insights from labels 
        insights = {"train_time": end - start}
        object_key = os.path.join(self.expt_dir, 
                                  'insights_{}.pkl'.format(self.cur_loop))
        
        self.client.write(labels, self.bucket_name, object_key, 'pickle')
        
        return 
     
    
    def test(self):
        ''' Test the performance of the model 
        write predictions and ground truth to the S3 
        bucket
        
        only write ground-truth to S3 once (cur_loop==0)
        
        Return:
        -------
            (predictions, ground_truth)
        '''
        res = self.test_fn(ckpt_file=self.ckpt_file, logdir=self.logdir)
        
        predictions, ground_truth = res['predictions'], res['labels']
        
        # write predictions and labels to S3
        object_key=os.path.join(self.expt_dir, 
                                "predictions_{}.pkl".format(self.cur_loop))
        self.client.write(predictions, self.bucket_name, object_key, 'pickle')
        
        if self.cur_loop==0:
            # write ground truth to S3
            object_key=os.path.join(self.expt_dir, 
                                    "ground_truth.pkl".format(self.cur_loop))
            self.client.write(ground_truth, self.bucket_name, object_key, 'pickle')
        
        self.compute_metrics(predictions, ground_truth)
        return 
    
    def compute_metrics(self, predictions, ground_truth):
        if self.type == 'Object Detection':
            from alectio_sdk.metrics.object_detection import Metrics, batch_to_numpy
            
            det_boxes, det_labels, det_scores, true_boxes, true_labels = batch_to_numpy(
                predictions, ground_truth)
        
            m = Metrics(det_boxes=det_boxes, 
                        det_labels=det_labels,
                        det_scores=det_scores, 
                        true_boxes=true_boxes,
                        true_labels=true_labels, 
                        num_classes=len(self.meta_data['class_labels']))
            
            
            metrics = {
                "mAP" : m.getmAP(),
                "AP" : m.getAP(),
                "precision" : m.getprecision(),
                "recall": m.getrecall(),
                "confusion_matrix" : m.getCM().tolist(),
                "class_labels" : self.meta_data['class_labels']
            }
            
        if self.type=="Image Classification" or self.type=="Text Classification":
            pass

        # save metrics to S3
        object_key = os.path.join(self.expt_dir, "metrics_{}.pkl".format(self.cur_loop))
        self.client.write(metrics, self.bucket_name, object_key, 'pickle')
        return
    
        
    def infer(self):
        ''' Infer on the unlabeled'''
        # Get unlabeled
        ts = range(self.meta_data['train_size'])
        self.unlabeled = list(set(ts) - set(self.labeled))
        
        outputs = self.infer_fn(unlabeled=deepcopy(self.unlabeled),
                           ckpt_file=self.ckpt_file, logdir=logdir)['outputs']
        
        # write the output to S3
        key = os.path.join(self.expt_dir, "outputs_{}.pkl".format(self.cur_loop))
        self.client.write(outputs, self.bucket_name, key, 'pickle')
        return
    

    def __call__(self, debug=False, host='0.0.0.0', port=5000):
        '''Run the app

        Paramters:
        ----------
        debug: boolean. Default: False
            If set to true, then the app runs in debug mode
            See https://flask.palletsprojects.com/en/1.1.x/api/#flask.Flask.debug

        host: the hostname to listen to. Default: '0.0.0.0'
            By default the app available to external world

        port: the port of the webserver. Default: 5000
        '''
        self.app.run(debug=debug, host=host, port=5000)






