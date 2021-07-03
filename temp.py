import torch
from torchvision import models
import numpy as np

print(torch.cuda.is_available())

image = np.random.random(size=[2, 3, 224, 224])

image_tensor = torch.from_numpy(image).float()
image_tensor = torch.autograd.Variable(image_tensor)
image_tensor = image_tensor.cuda()

model = models.resnet50(pretrained=False)
model = model.cuda()

out = model(image_tensor)
print(out)

# IPython.embed()
