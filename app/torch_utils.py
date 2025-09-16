import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


# Create the pipeline
# Load the model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


# Hyper-parameters
input_size = 784  # 28x28
hidden_size = 500
num_classes = 10


model = NeuralNet(input_size, hidden_size, num_classes)

FILE = "app/mnist_ffnn.pth"
# Load to  GPU
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load(FILE, map_location=device))
model.to(device)  # model on GPU or CPU
model.eval()  # for inference


# image --> tensor
def transform_image(image_bytes):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)


# prediction
def get_prediction(image_tensor):
    images = image_tensor.reshape(-1, 28 * 28).to(device)

    # Forward pass
    outputs = model(images)
    # Max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()
