# Form implementation generated from reading ui file 'imgreconition.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!
import json
import sys

import requests
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import urllib, urllib.request
import ssl

import base64

# 自己申请到的 图像文字识别 文字识别
API_KEY = ' '
SECRET = ' '

# 另外获取token值 图像识别
P_KEY = '  '
P_SECRET = '  '


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(597, 471)
        self.horizontalLayoutWidget = QtWidgets.QWidget(Form)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(90, 50, 171, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.comboBox = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout.addWidget(self.comboBox)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(Form)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(40, 100, 271, 31))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.lineEdit = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_2.addWidget(self.lineEdit)
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(40, 140, 271, 261))
        # self.label_3.setStyleSheet("background-color: rgb(255, 255, 127);")
        self.label_3.setStyleSheet("border-width: 1px;border-style: solid;boder-color: rgb(0,0,0);")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(330, 50, 241, 321))
        # self.label_4.setStyleSheet("background-color: rgb(85, 255, 0);")
        self.label_4.setStyleSheet("border-width: 1px;border-style: solid;boder-color: rgb(0,0,0);")
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(330, 380, 241, 23))
        self.pushButton_2.setObjectName("pushButton_2")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

'''
1.QFileDialog.getOpenFileName(self.horizontalLayoutWidget_2,"选择要识别的图片","/","Image File(*.jpg *.png)")
  getOpenFileName返回一个被用户选中的文件的路径，前提是这个文件是存在的。
  第一个参数parent，用于指定父组件。注意，很多Qt组件的构造函数都会有这么一个parent参数，并提供一个默认值0；
  第二个参数caption，是对话框的标题；
  第三个参数dir，是对话框显示时默认打开的目录，"." 代表程序运行目录，"/" 代表当前盘符的根目录(Windows，Linux下/就是根目录了)，也可以是平台相关的，比如"C:\\"等；例如我想打开程序运行目录下的Data文件夹作为默认打开路径，这里应该写成"./Data/"，若想有一个默认选中的文件，则在目录后添加文件名即可："./Data/teaser.graph"
  第四个参数filter，是对话框的后缀名过滤器，比如我们使用"Image Files(*.jpg *.png)"就让它只能显示后缀名是jpg或者png的文件。如果需要使用多个过滤器，使用";;"分割，比如"JPEG Files(*.jpg);;PNG Files(*.png)"；
  第五个参数selectedFilter，是默认选择的过滤器；
  第六个参数options，是对话框的一些参数设定，比如只显示文件夹等等，它的取值是enum QFileDialog::Option，每个选项可以使用 | 运算组合起来。
  如果我要想选择多个文件怎么办呢？Qt提供了getOpenFileNames()函数，其返回值是一个QStringList。你可以把它理解成一个只能存放QString的List，也就是STL中的list<string>。

2.QPixmap类用于绘图设备的图像显示，它可以作为一个QPaintDevice对象，也可以加载到一个控件中，通常是标签或按钮，用于在标签或按钮上显示图像。
  QPixmap可以读取的图像文件类型有BMP、GIF、JPG、JPEG、PNG、PBM、PGM、PPM、XBM、XPM等。
  利用QImage、QPxmap类可以实现图像的显示，并且利用类中的方法可以实现图像的基本操作(缩放、旋转)。

3.QApplication.clipboard()

4.完善运行测试代码，如下：
'''

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = QtWidgets.QMainWindow()      # 创建一个主窗体（必须要有一个主窗体）
    content = SimpleDialogForm()        # 创建对话框
    content.setupUi(main)               # 将对话框依附于主窗体
    main.show()                         # 主窗体显示
    sys.exit(app.exec_())
