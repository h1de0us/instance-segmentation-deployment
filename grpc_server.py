import grpc
from concurrent import futures
import inference_pb2
import inference_pb2_grpc

import torch
import requests
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from io import BytesIO
from PIL import Image
from torchvision.transforms.functional import to_tensor

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.75)
model.eval()

categories = weights.meta['categories']

def predict_image_objects(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image_tensor = to_tensor(image).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(image_tensor)
    
    object_names = [categories[i] for i in prediction[0]['labels'].tolist()]
    return object_names

class InstanceDetectorServicer(inference_pb2_grpc.InstanceDetectorServicer):
    def Predict(self, request, context):
        url = request.url
        objects = predict_image_objects(url)
        return inference_pb2.InstanceDetectorOutput(objects=objects)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InstanceDetectorServicer_to_server(InstanceDetectorServicer(), server)
    server.add_insecure_port('[::]:9090')
    server.start()
    print("gRPC server is running on port 9090.")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
