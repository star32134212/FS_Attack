import torch
from PIL import Image
import numpy as np
import samples.samples as DS

img_number = 300
DATASET = 'imagenet'
size = 224 #299 

ds = getattr(DS, DATASET)

indices = torch.randperm(len(ds.targets))[:img_number].numpy().astype(int) #隨機抽 img_number 張圖
images_list, labels = [np.array(ds[i][0].resize((size,size)).convert('RGB')) for i in indices], [np.array(ds[i][1]) for i in indices]
for i in range(len(images_list)):
    im = Image.fromarray(images_list[i])
    im.save(f"tmp_image/{i+1}.jpeg")