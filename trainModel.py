import sagemaker
import boto3
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch.estimator import PyTorch

sm_boto3 = boto3.client("sagemaker",region_name="eu-central-1")
sess = sagemaker.Session()
region = sess.boto_session.region_name
bucket = 'mlops.webapp' 
data_location  =  f's3://{bucket}'

input  =  { 
    'cfg' :  data_location + '/cfg' ,  
    #'weights': data_location+'/weights',  
    'images' :  data_location + '/images' ,  
    'labels' :  data_location + '/labels' } 

print(input)

hyperparameters = {'data': '/opt/ml/input/data/cfg/satellite-seg_new.yaml', 
                   'project': '/opt/ml/model/',
                   'name': 'satellite-seg',
                   'imgsz': 640,
                   'batch': 1,
                   'epochs': 2,
                   'workers':1}

metric_definitions = [{'Name': 'mAP50',
                       'Regex': '^all\s+(?:[\d.]+\s+){4}([\d.]+)'}]

yolo_estimator = PyTorch(entry_point='train.py',
                            source_dir='./src/',
                            role="{role}",
                            hyperparameters=hyperparameters,
                            framework_version='1.13.1',
                            py_version='py39',
                            script_mode=True,
                            instance_count=1,
                            metric_definitions=metric_definitions,
                            instance_type="ml.g4dn.2xlarge",
                            use_spot_instances = True,
                            max_wait = 7200,
                            max_run = 3600
            )

yolo_estimator.fit(input)
yolo_estimator.latest_training_job.wait(logs="None")
artifact = sm_boto3.describe_training_job(TrainingJobName=yolo_estimator.latest_training_job.name)["ModelArtifacts"]["S3ModelArtifacts"]
print("Model artifact persisted at " + artifact)