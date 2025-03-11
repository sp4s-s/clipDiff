import var
# ++++++++++++++++++++++++++++
import torch
from transformers import CLIPProcessor, CLIPModel
import requests
from PIL import Image

modelName = var.model_name # Model Nmae goes here or Model path 

model = CLIPModel.from_pretrained(modelName)
processor = CLIPProcessor.from_pretrained(modelName)

def ImageEncoder(image_path: str):
    image = Image.open(image_path)
    inputs = processor(images= image, return_tensors="pt")    
    
    with torch.no_grad():
        img_emb = model.get_image_features(**inputs)
    return img_emb


def TextEncoder(txt: str):
    inputs = processor(text=txt, return_tensors="pt", padding=True)
    with torch.no_grad():
        txt_features = model.get_text_features(**inputs)

    return txt_features

# Decode / Similarity 
def similarity(img_path: str, txt: str):
    img = ImageEncoder(img_path)
    txt = TextEncoder(txt)

    similarity = torch.cosine_similarity(img, txt)
    return similarity

def save_model(yes: bool):
    if yes == True:
        directory = "."
        model.save_pretrained(directory)
        processor.save_pretrained(directory)
        print("Model Saved")
    else:
        print("Not to be saved...!")










