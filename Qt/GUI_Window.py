import os
import time
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QGraphicsPixmapItem, QGraphicsScene
from PIL import Image
from package.net import Net
from package.load_model import load_model


class Ui_Classifier(object):
    def __init__(self):
        self.reco_class = None

    def setupUi(self, Classifier):
        Classifier.setObjectName("Classifier")
        Classifier.resize(1003, 877)
        font = QtGui.QFont()
        font.setPointSize(10)
        Classifier.setFont(font)
        self.centralwidget = QtWidgets.QWidget(Classifier)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(300, 0, 361, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(50, 50, 901, 521))
        self.graphicsView.setObjectName("graphicsView")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(50, 600, 171, 41))
        self.pushButton.setObjectName("pushButton")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(260, 600, 691, 41))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(260, 710, 691, 41))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(50, 710, 171, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(50, 780, 901, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(280, 660, 137, 31))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_2.setGeometry(QtCore.QRect(420, 660, 91, 27))
        self.radioButton_2.setObjectName("radioButton_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(80, 660, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        Classifier.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Classifier)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1003, 27))
        self.menubar.setObjectName("menubar")
        Classifier.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Classifier)
        self.statusbar.setObjectName("statusbar")
        Classifier.setStatusBar(self.statusbar)
        self.reco_class = None

        self.retranslateUi(Classifier)
        QtCore.QMetaObject.connectSlotsByName(Classifier)

    def retranslateUi(self, Classifier):
        _translate = QtCore.QCoreApplication.translate
        Classifier.setWindowTitle(_translate("Classifier", "MainWindow"))
        self.label.setText(_translate("Classifier", "Intelligent Classified Album"))
        self.pushButton.setText(_translate("Classifier", "Select Image"))
        self.pushButton_2.setText(_translate("Classifier", "Classify"))
        self.pushButton_3.setText(_translate("Classifier", "EXIT"))
        self.radioButton.setText(_translate("Classifier", "Vehicles"))
        self.radioButton_2.setText(_translate("Classifier", "Scenes"))
        self.label_2.setText(_translate("Classifier", "Select Class"))

    def choose_file(self):
        default_path = '/home/kana/Picture'
        if not os.path.exists(default_path):
            default_path = os.getcwd()
        dlg = QFileDialog(None, "choose_image_file", default_path, 'Image Files(*.png *.jpg *.jpeg)')
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            selected_name = dlg.selectedFiles()[0]
            if selected_name:
                self.lineEdit.setText(str(selected_name))

    def showImage(self):
        frame = QImage(self.lineEdit.text())
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.graphicsView.setScene(scene)

    def checkRadioButton(self):
        if self.radioButton.isChecked():
            self.reco_class = 'car'
        elif self.radioButton_2.isChecked():
            self.reco_class = 'scenes'

    def inf(self):
        if self.reco_class == 'car':
            classes = 10
        elif self.reco_class == 'scenes':
            classes = 5
        else:
            raise TypeError('Incorrect class selected.')

        inf_layers = [
            {'type': 'batchnorm', 'shape': (2, 224, 224, 3), 'affine': False, 'is_test': True},

            {'type': 'conv', 'shape': (8, 9, 9, 3)},
            {'type': 'batchnorm', 'shape': (216, 216, 8), 'is_test': True},
            {'type': 'relu'},

            {'type': 'maxpool', 'size': 4},

            {'type': 'conv', 'shape': (16, 5, 5, 8)},
            {'type': 'batchnorm', 'shape': (50, 50, 16), 'is_test': True},
            {'type': 'relu'},

            {'type': 'maxpool', 'size': 2},

            {'type': 'conv', 'shape': (32, 6, 6, 16)},
            {'type': 'batchnorm', 'shape': (20, 20, 32), 'is_test': True},
            {'type': 'relu'},

            {'type': 'maxpool', 'size': 2},

            {'type': 'transform', 'input_shape': (-1, 10, 10, 32), 'output_shape': (-1, 3200)},

            {'type': 'linear', 'shape': (3200, 64)},
            {'type': 'batchnorm', 'shape': (1, 64), 'is_test': True},
            {'type': 'relu'},

            {'type': 'dropout', 'drop_rate': 0.8, 'is_test': True},

            {'type': 'linear', 'shape': (64, classes)},
            {'type': 'batchnorm', 'shape': (2, classes), 'is_test': True},
            {'type': 'softmax'},
            {'type': 'infermean'}
        ]

        inf_net = Net(inf_layers)

        if self.reco_class == 'car':
            load_model(inf_net.parameters, '/home/kana/Huilanbei/CNN/saved_model/car/new/model_epoch4.npz')
            pass
        elif self.reco_class == 'scenes':
            load_model(inf_net.parameters, '/home/kana/Huilanbei/CNN/saved_model/scenes/train1/model_epoch2.npz')
        else:
            raise TypeError('Incorrect class selected.')

        x = Image.open(str(self.lineEdit.text()), mode='r').resize((224, 224))
        x = (np.asarray(x, dtype=float).copy() / 255).reshape(1, 224, 224, 3)

        start_time = time.time()
        output = inf_net.forward(x)
        y = np.argmax(output)
        end_time = time.time()
        cost_time = end_time - start_time
        Probability = np.max(output)

        if self.reco_class == 'car':
            if y == 0:
                result = "Bus"
            elif y == 1:
                result = "Family Sedan"
            elif y == 2:
                result = "Fire Engine"
            elif y == 3:
                result = "Heavy Truck"
            elif y == 4:
                result = "Jeep"
            elif y == 5:
                result = "Minibus"
            elif y == 6:
                result = "Racing Car"
            elif y == 7:
                result = "SUV"
            elif y == 8:
                result = "Taxi"
            elif y == 9:
                result = "truck"
            else:
                raise IndexError('Output label is incorrect.')
        elif self.reco_class == 'scenes':
            if y == 0:
                result = 'Church'
            elif y == 1:
                result = 'Desert'
            elif y == 2:
                result = 'Ice'
            elif y == 3:
                result = 'Lawn'
            elif y == 4:
                result = 'River'
            else:
                raise IndexError('Output label is incorrect.')
        else:
            raise TypeError('Incorrect class selected.')

        return result, Probability, cost_time

    def run_reco(self):
        if self.lineEdit.text() == '':
            self.lineEdit_2.setText(str('No Image Selected'))
        else:
            if not (self.radioButton.isChecked() or self.radioButton_2.isChecked()):
                self.lineEdit_2.setText(str('No Class Selected'))
            else:
                self.checkRadioButton()
                result, Probability, cost_time = self.inf()
                self.lineEdit_2.setText(str("Result: %s,   Probability: %.1f%%,   Cost %.3fs")
                                        % (result, Probability*100, cost_time))
