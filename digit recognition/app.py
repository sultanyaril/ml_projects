import cv2
import numpy as np
from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, LogSoftmax, Dropout
import torch

# Neural Network
class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class LeNet(Module):
    def __init__(self, num_channels, classes):
        super(LeNet, self).__init__()
        self.model = torch.nn.Sequential(
                  Conv2d(in_channels=num_channels, out_channels=20,
			  kernel_size=(5, 5)),
                  ReLU(),
                  MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                  Conv2d(in_channels=20, out_channels=50,
			kernel_size=(5, 5)),
                  ReLU(),
                  MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                  Flatten(),
                  Dropout(0.3),
                  Linear(in_features=800, out_features=500),
                  ReLU(),
                  Dropout(0.3),
                  Linear(in_features=500, out_features=classes),
                  LogSoftmax(dim=1)
        )
    def forward(self, x):
        return self.model(x)


model = LeNet(num_channels=1, classes=10)
model.load_state_dict(torch.load('trained_model_black_and_white.torch', map_location=torch.device('cpu')))
model.eval()

canvas = np.ones((600,600), dtype='uint8') * 255

canvas[100:500, 100:500] = 0

start_point=None
end_point=None
is_drawing=False


def draw_line(img, start_at, end_at):
    cv2.line(img, start_at, end_at, 255, 50)


def predict():
    image = canvas[100:500, 100:500]
    image = cv2.resize(image, dsize=(28, 28))
    canvas[:28, :28] = image
    image = torch.tensor(image, dtype=torch.float32)
    result = model(image.reshape(-1, 1, 28, 28)).argmax(dim=1)
    print('PREDICTION: ', result)


def on_mouse_events(event,x,y,flags,params):
    global start_point
    global end_point
    global canvas
    global is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
            is_drawing=True
            start_point=(x,y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            end_point=(x,y)
            draw_line(canvas,start_point,end_point)
            start_point=end_point
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing=False

cv2.namedWindow('Test Canvas')
cv2.setMouseCallback('Test Canvas', on_mouse_events)

while(True):
    cv2.imshow('Test Canvas', canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        is_drawing=True
    elif key == ord('c'):
        canvas[100:500, 100:500] = 0
    elif key == ord('p'):
        predict()

cv2.destroyAllWindows()

