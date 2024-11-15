import sagemaker
import boto3
from sagemaker.deserializers import JSONDeserializer
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.pytorch import PyTorchModel
from sagemaker.deserializers import JSONDeserializer

sm_boto3 = boto3.client("sagemaker",region_name="eu-central-1")
sess = sagemaker.Session()

model_data = 's3://mlops.webapp/results/pytorch-training-2023-11-30-19-33-54-618/output/model.tar.gz'
# model_data = 's3://sagemaker-eu-central-1-687462152766/pytorch-training-2023-11-30-23-09-47-893/output/model.tar.gz'
# model_data = 's3://mlops.webapp/model.tar.gz'

model = PyTorchModel(entry_point='inference.py',
                     source_dir='./src/',
                     model_data=model_data, 
                     framework_version='1.13.1', 
                     py_version='py39',
                     role="{role}",
                     env={'TS_MAX_RESPONSE_SIZE':'20000000'},
                     sagemaker_session=sess,
                     model_server_workers = 2)

predictor = model.deploy(initial_instance_count=1, 
                         instance_type="ml.g4dn.xlarge",
                         deserializer=JSONDeserializer(),
                         )
