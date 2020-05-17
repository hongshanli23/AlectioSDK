from ._object_detection import Metrics
import numpy as np

def batch_to_numpy(predictions, ground_truth):
    ''' batch predictions and ground-truth of the return 
    of test process
    
    Paramters:
    ----------
        predictions: dict
            predictions from the return of the test process
            for object detection
        
        ground_truth: dict
            ground-truth from the return of the test process
            for object detection
            
    
    Batch values predictions and ground-truth of detections
    on each image into a numpy array, and gather all 
    fields of predictions together into list
    
    Return:
    -------
        (det_boxes, det_labels, det_scores, true_boxes,  true_labels)
        See parameters for class ObjectDetection for the meaning of
        these parameters
    
    '''
    det_boxes, det_labels, det_scores = [], [], []
    true_boxes, true_labels = [], []
    for ix in predictions:
        det_boxes.append(np.array(predictions[ix]['boxes'], dtype=np.float32))
        det_labels.append(np.array(predictions[ix]['objects'], dtype=np.int32))
        det_scores.append(np.array(predictions[ix]['scores'], dtype=np.float32))
        
        true_boxes.append(np.array(ground_truth[ix]['boxes'], dtype=np.float32))
        true_labels.append(np.array(ground_truth[ix]['objects'], dtype=np.int32))
        
    return det_boxes, det_labels, det_scores, true_boxes, true_labels