# We'll now use the model and the trained weights to classify a sample image.

from PIL import Image
import torch
import torchvision.transforms as transforms
import model

# TODO: Make sure the path is correct
img_path = 'num.png'

weights_path = 'mnist_cnn_weights'

# Load the image and convert it to a tensor
img = Image.open(img_path).convert('L')
transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
img = transform(img).unsqueeze(0)

# Load the model and the trained weights
cnn = model.CNN()
cnn.load_state_dict(torch.load(weights_path, weights_only=True))

# Classify the image
output = cnn(img)
_, predicted = torch.max(output, 1)

print(f'The predicted number is: {predicted.item()}')