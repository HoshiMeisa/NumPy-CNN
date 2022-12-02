import sys
from functools import partial
from PyQt5.QtWidgets import QApplication, QMainWindow
from main_Window import Ui_Classifier


if __name__ == '__main__':
    app = QApplication(sys.argv)
    def close_window(): app.quit()
    MainWindow = QMainWindow()
    ui = Ui_Classifier()
    ui.setupUi(MainWindow)
    ui.pushButton.clicked.connect(partial(ui.choose_file))
    ui.pushButton.clicked.connect(partial(ui.showImage))
    ui.pushButton_2.clicked.connect(partial(ui.run_reco))
    ui.pushButton_3.clicked.connect(partial(close_window))
    MainWindow.show()
    sys.exit(app.exec_())
