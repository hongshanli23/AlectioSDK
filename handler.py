'''
This file includes end points for the platform backend to respond to 
call from frontend / client-side server / active learning engine
'''

import requests


    
def start_experiment():
    '''Handler for responding to the frontend call to start an experiment'''
    
    # get experiment_id and project_id
    # use it to fetch the experiment info from Dynamo
    # use this info to prepare a parameter dictionary `expt_params` 
    # to be send to the Active Learning Engine
    # see Parameters section in https://alectio.atlassian.net/wiki/spaces/PLATFORM/pages/249004033/ML+data+flow+architecture
    # for the expected key-values of the `expt_params`
    
    
    # use trigger compute_uncertainty dag from ALE with `expt_params` as the payload
    
    return



def resume_experiment():
    ''' Handler for responding to the frontend call to resume an experiment'''
    # this end point will be used once users finished labeling and resume the experiment 
    # from frontend
    
    # get experiment_id and project_id
    # use it to fetch experiment info from Dynamo
    
    # start one loop
    one_loop(client_ip, port, payload)

    # with the following payload
    '''
    {
      "cur_loop": <current loop>,
      "bucket_name": ...
      "user_id": ...
      "project_id": ...
      "experiment_id": ...
    }
    '''
    
    # expect a response {"status": 200} if client received the call
    # expect a response {"status": 500, "error": <error msg> } if the server on the client-side 
    # raised an exception
    return



def end_of_process_handler():
    ''' Respond to the calls from ALE
    At then end of each task from either ALE, 
    backend is expected to receive a call 
    
    Depending on the type of task, this handler 
    will respond differently. 
    
    Here is the list of task:
    
    'compute_uncertainty': The task for ALE to compute uncertainties
    'select': The task for ALE to select samples
    '''
    
    # payload <- Get payload for this call
    # payload looks like
    '''
    {
        "project_id": ...,
        "experiment_id": ...,
        "cur_loop":...
        "task": <what task just got compeleted>
        "status": <status of the task {"failed" or "completed"}
    }
    '''
    
    # project_info <- Use project_id to fetch project info from Dynamo
    # expt_info <- Use experiment_id to fetch experiment info from Dynamo
    
    # if the task is 'select':
        # This means ALE selected indices for the user 
        # i.e. users are using regular_al or auto_al
        # take a look at the field "fully_labeled" in `project_info`
        # it indicates if the data is fully labeled 
        
        # if the data is fully labeled
        # call CSS to train
        one_loop(client_ip, port, payload)
        
        # if the data is not fully labeled
        # do nothing 
        # when users resume the experiment, 
        # you will get a call from frontend to resume the expt
    
    # if the task is 'compute_uncertainty'
        # This means ALE finished computing uncertainty
        # This also means the users are using manual_al
        # do nothing
        # you will get a call from frontend to fetch uncertainty from S3


def end_loop_handler():
    '''what to do when CSS calls PBT that one loop is done'''
    # depending on the type of the experiment 
    # this handler behaves differently

    # payload <- Get payload of this call
    # use the payload to fetch experiment and project info

    # use the setting of the experiment decide to continue the loop or not
    
    # if continue:
        # call computee_uncertainty dag
    # else
        # pass

def one_loop(client_ip, port, payload):
    '''call client to train one loop'''
    
    # call client 
    # wait response
    
    # based on experiment setting
    # trigger dag or not
    pass


def trigger_dag(dag_name):
    pass
