from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import requests
from PIL import Image
from torchvision.transforms.functional import to_tensor
from starlette_prometheus import metrics
from io import BytesIO
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter

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


# setting everything up
app = FastAPI()
app.add_route("/metrics", metrics)
REQUESTS_COUNT = Counter("app_http_inference_count", "Multiprocess metric", ())


class ImageUrl(BaseModel):
    url: str

@app.post("/predict")
async def predict(image_url: ImageUrl):
    REQUESTS_COUNT.inc()
    try:
        objects = predict_image_objects(image_url.url)
        return {"objects": objects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)