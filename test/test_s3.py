from alectio_sdk.flask_wrapper.s3_client import S3Client

def test_write_read():
    bucket_name='alectio-sandbox'
    object_key = 'test/selected_indices'
    
    client = S3Client()
    
    obj = [i for i in range(10)]
    
    client.write(obj, bucket_name, object_key, 'pickle')
    
    r = client.read(bucket_name, object_key, 'pickle')
    
    assert obj == r