# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:08:32 2020

@author: arun
"""

from .imports import *

class CM(object):
    """
    Description:
        Confusion Matix skeleton that carries out basic confusion matrix operations 
        to obtain final confusion matrix
    
    nClasses - number of classes 
    labels   - target labels
    task     - specifies the target task (Object Detection/ Classification) 
    
    """
    def __init__(self,nClasses ,labels= None , task = 'OD'):
        self.nClasses  =  nClasses
        if task =='OD':
            self.CMClasses = self.nClasses+1                                       #### Add 1 to represent nothingness as an object
        else:
            self.CMClasses = self.nClasses
        self.CM        = np.zeros((self.CMClasses, self.CMClasses), dtype = np.float)
        if not labels:
            self.labels = list(range(self.CMClasses))
        else:
            self.labels = labels
    
    def updateCM(self, updatevals):
        
        """
        Desription:
            Function updates true positives, false positives and False negatives
        Arguments:
            Update vals - Zip containing tp,fp,fn values and indices
    
        """
        for ix , val in updatevals:
            self.CM[ix[0],ix[1]] = val
    
    def updateincorrectpredictions(self, currCM):
        """
        Description:
            Function adds on to the existing confusion matrix
        """
        
        self.CM+= currCM
        
    
    def save_CM(self,outdir,modelname):
        """
        Description:
            Fucntion saves Confusion matrix in target folder
        
        """
        df_cm = pd.DataFrame(self.CM,index =self.labels,columns= self.labels)
        df_cm.to_csv(os.path.join(outdir,str(modelname)+'confusion_matrix'+'.csv'),index=True)

    
    
    
    
        
        
        
        
        
        
        
        
        