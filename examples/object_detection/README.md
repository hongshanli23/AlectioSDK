copy samples of data from Alectio resources S3 bucket to your local machine
```
aws s3
```

inference data
take positive priors predicted by the model
apply non-maximum suppression

send back the the boxes info after NMS
```python
{
    img_i : {
        box_i : {
            "cooridnate" : [...],
            "class_prediction" : [...] # output for classes, do not apply softmax
            "objectness": [.] # between 0 and 1
            
        }
    }
}
```