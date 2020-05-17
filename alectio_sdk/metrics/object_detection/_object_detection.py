# -*- coding: utf-8 -*-
"""
Created on Fri May 15 23:19:25 2020

@author: arun
"""


from .utils import *
from .imports import *
from .CM import CM as CM


class Metrics(object):
    """
    Evaluation Metrics for object detection 

       Parameters:
       -----------
            det_boxes: list of numpy.ndarray, 
                one array for each image containing detected objects' bounding boxes.
                Each array is of shape (M_i, 4), where M_i is the number of objects 
                in the image. Bounding boxes are in xyxy-convention
                All bounding boxes are normalized according to image dimension

            det_labels: list of numpy.ndarray, 
                one array for each image containing detected objects' labels. 
                Each array is of shape (M_i)

            det_scores: list of numpy.ndarray, 
                one array for each image 
                containing detected objects' labels' scores.
                Each array is of shape (M_i). Object score is 
                the objectness of each detection


            true_boxes: list of numpy.ndarray, 
                one array for each image containing actual objects' bounding boxes.
                Each array is of shape (N_i, 4), where N_i is the number of 
                ground-truth bbox on the image.
                Bbox in xyxy convention; each bounding box is normalized according 
                to the image dimension

            true_labels: list of numpy.ndarray, 
                one array for each image containing actual objects' labels. 
                Each tensor is of shape N_i
            
            num_classes: int
                number of classes (not include background)
                    
            threshold: float. Default=0.5
                IoU threshold for a prediction to be considered correct
                with grounth-truth bouding box

    """
    
    def __init__(self,det_boxes, det_labels, det_scores, true_boxes, 
                 true_labels, num_classes,  threshold=0.5):
        
        self.det_boxes    = det_boxes
        self.det_labels   = det_labels
        self.det_scores   = det_scores
        self.true_boxes   = true_boxes
        self.true_labels  = true_labels
        self.num_classes  = num_classes
        self.threshold    = threshold
 
        self.reordered_trueboxes  = []
        self.reordered_detboxes   = []
        self.reordered_detscores  = []
        self.confusionMatrix      =  CM(self.num_classes)
        self.AP ={}
        self.recall = {}
        self.precision = {}
        self._reformatboxes()
        self._evaluate()
    
    def _reformatboxes(self):
        """
        Description:
            Reformats detection and annotation boxes labelwise 
        """
        for i in range(len(self.true_boxes)):
            self.reordered_trueboxes.append([np.array([]) for _ in range(self.num_classes)])
            self.reordered_detboxes.append([np.array([]) for _ in range(self.num_classes)])
            self.reordered_detscores.append([np.array([]) for _ in range(self.num_classes)])
            currtrueBoxes  = self.true_boxes[i]
            currtrueLabels = self.true_labels[i]
            
            ######## Map Annotation boxes labelwise ####################
            for label in range(self.num_classes):
                self.reordered_trueboxes[-1][label] = currtrueBoxes[currtrueLabels == label]
                
            if self.det_boxes[i] is not None:
                currdetSort   = np.argsort(self.det_scores[i])
                currdetLabels = self.det_labels[i][currdetSort]
                currdetBoxes  = self.det_boxes[i][currdetSort]
                currdetScores  = self.det_scores[i][currdetSort]
                               
                ######## Map Detection boxes labelwise ####################
                for label in range(self.num_classes):
                    self.reordered_detboxes[-1][label]  = currdetBoxes[currdetLabels == label]
                    self.reordered_detscores[-1][label] = currdetScores[currdetLabels == label]
                        
    def _evaluate(self):
        """
        Description:
            Computes AP,Precision and recall metrics per class for current model
        
        """
        for label in range(self.num_classes):
            tpBoolmap = detScores =[]
            nothingness = False
            tp = fp = num_annotations = 0
            for i in range(len(self.reordered_trueboxes)):
                currGT     = self.reordered_trueboxes[i][label]
                currDet    = self.reordered_detboxes[i][label]
                currDetscores = self.reordered_detscores[i][label]
                detectedGT = []
                num_annotations += currGT.shape[0]
                
                for box,score in zip(currDet,currDetscores):
                    detScores.append(score)
                    if currGT.shape[0] == 0:
                        tpBoolmap.append(0)
                        fp +=1
                        continue
                    Ious       = compute_iou(np.expand_dims(box, axis=0), currGT)
                    assignedGT = np.argmax(Ious, axis = 1)                         ##### Get the box with maximum overlap
                    currIou = Ious[0, assignedGT]
                    
                    if currIou >= self.threshold and assignedGT not in detectedGT:
                        tp+=1
                        detectedGT.append(assignedGT)
                        tpBoolmap.append(1)
                    else:
                        fp+=1
                        tpBoolmap.append(0)
                ############## Recompute confusion Matrix using current values ###############
                if i == (len(self.reordered_trueboxes) -1):
                    nothingness = True
                self.recompute_ConfusionMatrix(tp,fp,num_annotations,label,currDet,self.reordered_trueboxes[i],nothingness)
            
            if num_annotations == 0:
                self.AP[label] = 0
                continue
            tpBoolmap = np.array(tpBoolmap)
            fpBoolmap = np.ones_like(tpBoolmap) - tpBoolmap
            ix        = np.argsort(np.array(detScores))
            tpBoolmap = tpBoolmap[ix]
            fpBoolmap = fpBoolmap[ix]
            falsePositives = np.cumsum(fpBoolmap)
            truePositives  = np.cumsum(tpBoolmap)
            recall = self.recall(truePositives,num_annotations)
            precision = self.precision(truePositives,falsePositives)
            
            self.AP[label]     = compute_ap(recall, precision)
            self.recall[label] = np.sum(tpBoolmap)/num_annotations
            self.precision[label] = np.sum(tpBoolmap)/(np.sum(tpBoolmap)+np.sum(fpBoolmap))

    def getAP(self):
        """
        Returns:
            AP per class as a dictionary in the format {class: AP}
        """
        AP = {}
        for c in self.AP:
            AP[c] = self.AP[c].item()
        return AP
    
        
    def getmAP(self):
        """
        Description:
            Computes mAP 
        
        Returns:
             mAP score as float
        """
        return np.mean([ap for c, ap in self.AP.items()]).item()
    
    def getprecision(self):
        """
        Description:
            Computes Overall Precision 
        
        Returns:
             Precision score as float
        """
        return np.mean([prec for c, prec in self.precision.items()]).item()
    
    def getrecall(self):
        """
        Description:
            Computes recall
        
        Returns:
             Recall score as float
        """
        return np.mean([rec for c, rec in self.recall.items()]).item()
    
    
    def process_incorrect(self,detections, GTannotations , currlabel):
        """
        Description:
            Function processes detection boxes to count mispredictions and 
            update the confusion matrix
        Arguments:
        detections    - Detection boxes of currentlabel 
        GTannotations - Ground Truth annotations of current image 
        currlabel     - Label currently being processed
        
        Returns:
            Currently recomputed incorrect detections confusion matrix
        """
        currCM = np.zeros_like(self.confusionMatrix.CM)
        for label in range(self.num_classes):
            if label == currlabel:
                continue
            annotations = GTannotations[label]
            if annotations.shape[0] == 0:
                continue
            detected_annotations = []
            for box in detections:
                Ious = compute_iou(np.expand_dims(box, axis=0), annotations)
                assignedGT = np.argmax(Ious, axis=1)
                maxIou = Ious[0, assignedGT]
                if maxIou >= self.threshold and assignedGT not in detected_annotations:
                    detected_annotations.append(assignedGT)
                    currCM[label, currlabel] +=1
        return currCM
        
    def precision(self,truePositives, falsePositives):
        """
        Description:
            Fuction computes precision
        Inputs:
            truePositives - True Positives envelope of current label
            falsePositives - False Positives envelope of current label
        
        Returns:
            Precision metrics as float
        """
        return truePositives / np.maximum(truePositives + falsePositives, np.finfo(np.float64).eps)
        
    def recall(self,truePositives, nGT):
        """
        Description:
            Fuction computes recall
        Inputs:
            truePositives - True Positives envelope of current label
            nGT - Number of Ground truth annotations
        
        Returns:
            Recall metrics as float
        """
        return (truePositives/nGT)
    

    def recompute_ConfusionMatrix(self,tp,fp,num_annotations,label,currDet,currTrue,nothingness = False):
        
        """
        Description:
            Recomputes confusion matrix imagewise
        Inputs:
            tp - number of True positives
            fp - number of False positives
            num_annotations - number of current ground truth annotations processed
            label    - current label being processed
            currDet  - current detections being processed
            currTrue - Ground truth boxes for current image
        
        Returns:
            Recall metrics as float
        """
        ############## Update current values into confusion Matrix ###############
        if nothingness:
            fn  = num_annotations - tp
            Updatesvals = zip([[label,label], [-1,label] , [label,-1]] , [tp,fp,fn])
            self.confusionMatrix.updateCM(Updatesvals)
        currCM = self.process_incorrect(currDet, currTrue , label)
        self.confusionMatrix.updateincorrectpredictions(currCM)
        
    def getCM(self):
        """
        Description:
            Function returns computed Confusion matrix
        returns:
            Computed Confusion Matrix
        
        """
        return self.confusionMatrix.CM
        
