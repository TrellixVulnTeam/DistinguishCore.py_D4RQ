from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtGui import QPixmap, QPainter
from ui.untitled import Ui_Form
from ui.pic import Ui_Form1
from DistinguishCore import MainUI
from UserDataRecord import UserDataRecordUI
from UserDataManage import UserDataManageUI
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ConnectAll(MainUI, QMainWindow):
    def __init__(self):
        super(ConnectAll, self).__init__()
        style = open(os.path.join('all.qss')).read()
        self.setStyleSheet(style)

        #self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.startWebcamButton.setStyleSheet(
            "startWebcamButton{color:black}"
            "startWebcamButton:hover{color:red}"
            "startWebcamButton{background-color:rgb(180,180,180)}"
            "startWebcamButton{border:2px}"
            "startWebcamButton{border-radius:10px}"
            "startWebcamButton{padding:2px 4px}"
            "startWebcamButton{font-size:14pt}")

    def mouseMoveEvent(self, e: QMouseEvent):
        self._endPos = e.pos() - self._startPos
        self.move(self.pos() + self._endPos)

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._isTracking = True
            self._startPos = QPoint(e.x(), e.y())

    def mouseReleaseEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._isTracking = False
            self._startPos = None
            self._endPos = None

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        QPainter(self).drawPixmap(self.rect(), QPixmap("./背景图.png"))

    def closewin(self):
        self.close()


class Window1(UserDataRecordUI, QMainWindow):
    def __init__(self):
        super(Window1, self).__init__()
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        style = open(os.path.join('all.qss')).read()
        self.setStyleSheet(style)

    def OPEN(self):
        self.show()

    def closewin(self):
        self.close()

    def returnmain(self):
        self.pushButton.clicked.connect(main.show)
        self.pushButton.clicked.connect(win1.hide)


class Window2(UserDataManageUI, QMainWindow):
    def __init__(self):
        super(Window2, self).__init__()
        self.setStyleSheet(open(os.path.join('all.qss')).read())
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)

    def OPEN(self):
        self.show()

    def closewin(self):
        self.close()

    def returnmain(self):
        self.pushButton.clicked.connect(main.show)
        self.pushButton.clicked.connect(win2.hide)


class HelpWindow(Ui_Form, QWidget):
    def __init__(self):
        super(HelpWindow, self).__init__()
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setStyleSheet(open(os.path.join('all.qss')).read())
        self.setupUi(self)

    def OPEN(self):
        self.show()

    def returnmain(self):
        self.pushButton.clicked.connect(main.show)
        self.pushButton.clicked.connect(helpWindow.hide)

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        QPainter(self).drawPixmap(self.rect(), QPixmap("./背景图.png"))


class AddWindow(Ui_Form1, QWidget):
    def __init__(self):
        super(AddWindow, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.close)
        self.setStyleSheet(open(os.path.join('all.qss')).read())
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)

    def OPEN(self):
        self.show()

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        QPainter(self).drawPixmap(self.rect(), QPixmap("./背景图.png"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = ConnectAll()
    win1 = Window1()
    win2 = Window2()
    helpWindow = HelpWindow()
    addWindow = AddWindow()

    main.stackedWidget.addWidget(win1)
    main.stackedWidget.addWidget(win2)
    main.stackedWidget.setCurrentIndex(0)

    main.show()

    main.pushButton2.clicked.connect( 
        lambda: main.stackedWidget.setCurrentIndex(0))
    main.pushButton.clicked.connect( 
        lambda: main.stackedWidget.setCurrentIndex(1))
    main.pushButton_2.clicked.connect( 
        lambda: main.stackedWidget.setCurrentIndex(2)) 

    main.pushButton_3.clicked.connect(main.hide)
    main.pushButton_3.clicked.connect(helpWindow.OPEN)
    #main.pushButton_11.clicked.connect(addWindow.OPEN)
    helpWindow.pushButton.clicked.connect(helpWindow.returnmain)

    sys.exit(app.exec_())
