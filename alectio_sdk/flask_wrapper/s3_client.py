import boto3
import os
import pickle
import json

class S3Client:
    '''Boto3 client to S3'''
    def __init__(self):
    
        # boto3 clients to read / write to S3 bucket
        self.client = boto3.client(
                's3', 
                aws_access_key_id=os.getenv('ACCESS_KEY'),
                aws_secret_access_key=os.getenv('SECRET_KEY')
                )
        
    
    def read(self, bucket_name, object_key, file_format):
        ''' Read a file from the S3 bucket containing
        this experiment
    
        object_key: str.
           object key for this file
           
        file_format: str.
            format of the file {"pickle", "json"}
        
        '''
            
        s3_object=self.client.get_object(
            Bucket=bucket_name, Key=object_key)
        body = s3_object['Body']

        if file_format=='json':
            jstr=body.read().decode('utf-8')
            content = json.loads(jstr)
        elif file_format=='pickle':
            f = body.read()
            content = pickle.loads(f)    
        elif file_format=='txt':
            content = body.read().decode(encoding="utf-8", errors="ignore")
        return content
    
    
    def write(self, obj, bucket_name, object_key, file_format):
        '''Write an object to S3 bucket
        Mostly used for writing ExperimentData.pkl
        InferenceData.pkl files 
        
        obj: dict | list | string
        
        bucket_name: name of the s3 bucket
        object_key: str.
            object key in the S3 bucket
            
        file_format: str.
            format of the file to save the object 
            {pickle, json}

        '''
        
        # convert obj to byte string
        if file_format=='pickle':
            bytestr = pickle.dumps(obj)
        elif file_format=='json':
            bytestr = json.dumps(obj)
        elif file_format=='txt':
            bytestr = b'{}'.format(string)
            
        
        # @TODO add md5 hash 
        # @TODO return success or failure message
        # put in S3
        r = self.client.put_object(
            Bucket=bucket_name, 
            Key=object_key,
            Body=bytestr,
        )
        
        return

