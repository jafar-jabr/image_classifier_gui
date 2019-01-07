#!/usr/bin/env python

"""
@author: Jafar Jabr <jafaronly@yahoo.com>
=======================================
"""
from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout, QApplication, QFileDialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import json
import numpy as np
from PIL import Image
from collections import OrderedDict
import torch
from torch import nn
from torchvision import models


class DialogResnet(QDialog):
    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle("FLOWERS TYPE PREDICTOR")
        self.show_file_browser = True
        self.img_url = ''
        self.model = self.load_checkpoint('classifier.pth')
        # self.model = self.load_checkpoint('classifier987775.pt')
        self.selected_model = 1
        layout = QVBoxLayout(self)
        self.figure = plt.figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        btn1 = QPushButton('Input Image')
        btn1.clicked.connect(lambda me: self.handle_main_btn())
        layout.addWidget(btn1)

        btn2 = QPushButton('Let me guess')
        btn2.clicked.connect(lambda me: self.try_to_predict())
        layout.addWidget(btn2)

        self.resize(1120, 800)
        self.initial_show_image()
        self.device = "cpu"

    def handle_main_btn(self):
        if self.show_file_browser:
            self.open_file()
        else:
            self.initial_show_image(self.img_url)
            self.show_file_browser = True

    def try_to_predict(self):
        image_path = self.img_url
        self.plot_solution(image_path, self.model)

    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        img_path, _ = QFileDialog.getOpenFileName(None, "choose an image", "", "Image Files (*.jpg *.png)", options=options)
        if img_path:
            self.img_url = img_path
            self.initial_show_image(img_path)

    @staticmethod
    def load_checkpoint(checkpoint_path):
        model = models.resnet152(pretrained=True)
        layers = [
            ('dropout', nn.Dropout(0.2)),
            ('fc1', nn.Linear(2048, 512)),
            ('relu1', nn.ReLU()),
            ('dropout2', nn.Dropout(0.2)),
            ('fc2', nn.Linear(512, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]
        classifier = nn.Sequential(OrderedDict(layers))
        chpt = torch.load(checkpoint_path, map_location='cpu')
        model.fc = classifier
        model.class_to_idx = chpt["class_to_idx"]
        model.load_state_dict(chpt["state_dict"], strict=False)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        return model

    def calc_accuracy(self, model, data):
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(self.dataloaders[data]):
                # obtain the outputs from the model
                outputs = model.forward(inputs)
                outputs = outputs.to(self.device)
                # max provides the (maximum probability, max value)
                _, predicted = outputs.max(dim=1)
                # check the
                if idx == 0:
                    print(predicted)  # the predicted class
                    print(torch.exp(_))  # the predicted probability
                equals = predicted == labels.data
                if idx == 0:
                    print(equals)
                print(equals.float().mean())

    @staticmethod
    def process_image(image_path):
        '''
        Scales, crops, and normalizes a PIL image for a PyTorch
        model, returns an Numpy array
        '''
        # Open the image
        img = Image.open(image_path)
        # Resize
        if img.size[0] > img.size[1]:
            img.thumbnail((10000, 256))
        else:
            img.thumbnail((256, 10000))
        # Crop
        left_margin = (img.width - 224) / 2
        bottom_margin = (img.height - 224) / 2
        right_margin = left_margin + 224
        top_margin = bottom_margin + 224
        img = img.crop((left_margin, bottom_margin, right_margin,
                        top_margin))
        # Normalize
        img = np.array(img) / 255
        mean = np.array([0.485, 0.456, 0.406])  # provided mean
        std = np.array([0.229, 0.224, 0.225])  # provided std
        img = (img - mean) / std

        # Move color channels to first dimension as expected by PyTorch
        img = img.transpose((2, 0, 1))

        return img

    @staticmethod
    def imshow(image, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots()
        if title:
            ax.set_title(title)
        # PyTorch tensors assume the color channel is first
        # but matplotlib assumes is the third dimension
        image = image.transpose((1, 2, 0))

        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        # Image needs to be clipped between 0 and 1
        image = np.clip(image, 0, 1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(image)
        # plt.show()
        return ax

    def initial_show_image(self, img_path="images/no_image.jpg"):
        plt.close('all')
        self.figure.clear()
        # create an axis
        ax = self.figure.add_subplot(111)
        ax.set_title('Believe it or not, I know 102 types of flowers, more than what you know ^_*')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        # discards the old graph
        dd = plt.imread(img_path)
        # plot data
        ax.imshow(dd)
        # refresh canvas
        self.canvas.draw()

    def predict(self, image_path, model, top_num=5):
        # Process image
        img = self.process_image(image_path)
        cat_to_name = json.load(open('cat_to_name.json'))
        image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
        # Add batch of size 1 to image
        model_input = image_tensor.unsqueeze(0)
        # Probs
        probs = torch.exp(model.forward(model_input))

        # Top probs
        top_probs, top_labs = probs.topk(top_num)
        top_probs = top_probs.detach().cpu().numpy().tolist()[0]
        top_labs = top_labs.detach().cpu().numpy().tolist()[0]

        # Convert indices to classes
        idx_to_class = {val: key for key, val in
                     model.class_to_idx.items()}
        top_labels = [idx_to_class[lab] for lab in top_labs]
        top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
        return top_probs, top_labels, top_flowers

    def plot_solution(self, image_path, model):
            # Make prediction
            probs, labs, flowers = self.predict(image_path, model)
            #clear old results
            plt.rcdefaults()
            plt.close('all')
            self.figure.clear()
            # Set up plot
            ax = self.figure.add_subplot(2, 1, 1)
            # Set up title
            title_ = flowers[0]
            # Plot flower
            img = self.process_image(image_path)
            self.imshow(img, ax, title=title_)
            # Plot bar chart
            ax2 = self.figure.add_subplot(2, 1, 2)
            ax2.set_xlabel("Probabilities")
            y_pos = [x for x in range(len(flowers))]
            # y_pos = np.arange(len(flowers))
            ax2.barh(y_pos, probs, align='center', color='blue', linewidth=0)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(flowers)
            ax2.invert_yaxis()
            self.canvas.draw()


if __name__ == '__main__':
    app = QApplication([])
    my_dialog = DialogResnet(None)
    my_dialog.exec_()
