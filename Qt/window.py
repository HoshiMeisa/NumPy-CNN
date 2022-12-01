# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Classifier(object):
    def setupUi(self, Classifier):
        Classifier.setObjectName("Classifier")
        Classifier.resize(794, 679)
        font = QtGui.QFont()
        font.setPointSize(10)
        Classifier.setFont(font)
        self.centralwidget = QtWidgets.QWidget(Classifier)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(280, 10, 201, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(70, 70, 651, 431))
        self.graphicsView.setObjectName("graphicsView")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(70, 520, 108, 41))
        self.pushButton.setObjectName("pushButton")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(190, 520, 531, 41))
        self.lineEdit.setObjectName("lineEdit")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(90, 580, 71, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(190, 580, 531, 41))
        self.lineEdit_2.setObjectName("lineEdit_2")
        Classifier.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Classifier)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 794, 27))
        self.menubar.setObjectName("menubar")
        Classifier.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Classifier)
        self.statusbar.setObjectName("statusbar")
        Classifier.setStatusBar(self.statusbar)

        self.retranslateUi(Classifier)
        self.pushButton.clicked.connect(Classifier.click1) # type: ignore
        self.graphicsView.rubberBandChanged['QRect','QPointF','QPointF'].connect(Classifier.show_pic) # type: ignore
        self.lineEdit.textEdited['QString'].connect(Classifier.show_path) # type: ignore
        self.lineEdit_2.textEdited['QString'].connect(Classifier.show_result) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Classifier)

    def retranslateUi(self, Classifier):
        _translate = QtCore.QCoreApplication.translate
        Classifier.setWindowTitle(_translate("Classifier", "MainWindow"))
        self.label.setText(_translate("Classifier", "智能分类相册"))
        self.pushButton.setText(_translate("Classifier", "选择图片"))
        self.label_2.setText(_translate("Classifier", "分类结果"))
