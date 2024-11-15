

import boto3, cv2, time, numpy as np, matplotlib.pyplot as plt, random
from sagemaker.pytorch import PyTorchPredictor
from sagemaker.deserializers import JSONDeserializer
import requests

sm_client = boto3.client(service_name="sagemaker",region_name="eu-central-1")

ENDPOINT_NAME = 'pytorch-inference-2023-12-14-19-04-15-422'
print(f'Endpoint Name: {ENDPOINT_NAME}')

endpoint_created = False
while True:
    response = sm_client.list_endpoints()
    for ep in response['Endpoints']:
        print(f"Endpoint Status = {ep['EndpointStatus']}")
        if ep['EndpointName']==ENDPOINT_NAME and ep['EndpointStatus']=='InService':
            endpoint_created = True
            break
    if endpoint_created:
        break
    time.sleep(5)
    
predictor = PyTorchPredictor(endpoint_name=ENDPOINT_NAME,
                             deserializer=JSONDeserializer())

infer_start_time = time.time()

# orig_image = cv2.imread('dataset/images/train/u3s4k1hylelkfcx9ts_output_183.jpg')
# orig_image = cv2.imread('u3s4k1hylelkfcx9s7_output_196.jpg')
response = requests.get("https://www.neuraldesigner.com/images/tree-wilt.webp")
orig_image = cv2.imdecode(np.frombuffer(response.content, np.uint8), 1)

image_height, image_width, _ = orig_image.shape
model_height, model_width = 300, 300
x_ratio = image_width/model_width
y_ratio = image_height/model_height

resized_image = cv2.resize(orig_image, (model_height, model_width))
payload = cv2.imencode('.jpg', resized_image)[1].tobytes()
result = predictor.predict(payload)
# print("result: ", result['boxes'])
infer_end_time = time.time()

print(f"Inference Time = {infer_end_time - infer_start_time:0.4f} seconds")

if 'boxes' in result:
    for idx,(x1,y1,x2,y2,conf,lbl) in enumerate(result['boxes']):
        # Draw Bounding Boxes
        x1, x2 = int(x_ratio*x1), int(x_ratio*x2)
        y1, y2 = int(y_ratio*y1), int(y_ratio*y2)
        color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
        cv2.rectangle(orig_image, (x1,y1), (x2,y2), color, 4)
        cv2.putText(orig_image, f"Class: {int(lbl)}", (x1,y1-40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(orig_image, f"Conf: {int(conf*100)}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        if 'masks' in result:
            # Draw Masks
            mask = cv2.resize(np.asarray(result['masks'][idx]), dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            for c in range(3):
                orig_image[:,:,c] = np.where(mask>0.5, orig_image[:,:,c]*(0.5)+0.5*color[c], orig_image[:,:,c])

if 'probs' in result:
    # Find Class
    lbl = result['probs'].index(max(result['probs']))
    color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    cv2.putText(orig_image, f"Class: {int(lbl)}", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
if 'keypoints' in result:
    # Define the colors for the keypoints and lines
    keypoint_color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    line_color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))

    # Define the keypoints and the lines to draw
    # keypoints = keypoints_array[:, :, :2]  # Ignore the visibility values
    lines = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Torso
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]

    # Draw the keypoints and the lines on the image
    for keypoints_instance in result['keypoints']:
        # Draw the keypoints
        for keypoint in keypoints_instance:
            if keypoint[2] == 0:  # If the keypoint is not visible, skip it
                continue
            cv2.circle(orig_image, (int(x_ratio*keypoint[:2][0]),int(y_ratio*keypoint[:2][1])), radius=5, color=keypoint_color, thickness=-1)

        # Draw the lines
        for line in lines:
            start_keypoint = keypoints_instance[line[0]]
            end_keypoint = keypoints_instance[line[1]]
            if start_keypoint[2] == 0 or end_keypoint[2] == 0:  # If any of the keypoints is not visible, skip the line
                continue
            cv2.line(orig_image, (int(x_ratio*start_keypoint[:2][0]),int(y_ratio*start_keypoint[:2][1])),(int(x_ratio*end_keypoint[:2][0]),int(y_ratio*end_keypoint[:2][1])), color=line_color, thickness=2)

plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
plt.show()