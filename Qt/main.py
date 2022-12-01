import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore
from functools import partial
import conver_ui


def cul_convert(ui):
    input = ui.lineEdit.text()
    result = float(input) * 6.7
    ui.lineEdit_2.setText(str(result))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = conver_ui.Ui_A_Simple_Convertor()
    ui.setupUi(MainWindow)
    ui.pushButton.clicked.connect(partial(cul_convert, ui))
    ui.pushButton_2.clicked.connect
    MainWindow.show()
    sys.exit(app.exec_())



