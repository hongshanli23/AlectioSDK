from alectio_sdk.flask_wrapper import Pipeline
from alectio_sdk.flask_wrapper.s3_client import S3Client



import os


def test_obj_detect_pipeline():
    # test Pipeline
    
    
    def train_fn(labeled, resume_from, ckpt_file, logdir):
        return {i: 0 for i in range(10)}
    
    def test_fn(ckpt_file, logdir):
        predictions = {
            0 : {
                "boxes": [[0.1, 0.2, 0.3, 0.4]],
                "objects" : [1],
                "scores" : [0.7]
            }
        }
        
        lbs = {
            0 : {
                "boxes": [0.2, 0.3, 0.4, 0.5],
                "objects": [1],
                
            }
        }
        
        return {"predictions": predictions, "labels": lbs}
        
    def infer_fn(unlabeled, ckpt_file, logdir):
        outputs = {
            0 : {
                "boxes" : [[0.2, 0.4, 0.7, 0.5]],
                "pre_softmax": [[7.4, 8.2]],
                "scores": [0.6]
                
            }
        }
        return {"ouputs": outputs}
    
    
    payload  = {
        "bucket_name" : "alectio-sandbox",
        "user_id": "test_user",
        "project_id" : "test_proj",
        "experiment_id": "test_expt",
        "cur_loop": 3,
        "type": "Object Detection"
    
        }
    
    client = S3Client()
    
    # generate some stuff to S3
    for i in range(payload['cur_loop']+1):
        obj = [j for j in range(i*10, i*10+10)]
        key = os.path.join(payload['user_id'],
                           payload['project_id'],
                           payload['experiment_id'],
                           "selected_indices_{}.pkl".format(i))
        
        client.write(obj, payload['bucket_name'], 
                    key, 'pickle')
        
    # generate meta-data 
    meta = {
        "train_size": 1,
        "test_size": 1,
        "class_labels":{
            0: 'car',
            1: 'ped'
        }
    }
    
    key = os.path.join(payload['user_id'], 
                       payload['project_id'], 
                       "meta.json")
    
    client.write(meta, payload['bucket_name'], key, 'json')
    

    pipeline = Pipeline('test', train_fn, test_fn, infer_fn)
    
    pipeline._one_loop(payload)