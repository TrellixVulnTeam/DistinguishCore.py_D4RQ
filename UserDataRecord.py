from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QRegExp, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon, QRegExpValidator, QTextCursor, QPainter
from PyQt5.QtWidgets import QDialog, QApplication, QWidget, QMessageBox
from PyQt5.uic import loadUi
import os
import sys
import cv2
import queue
import logging
import logging.config
from datetime import datetime
import threading
import sqlite3

class UserDataRecordUI(QWidget):
    receiveLogSignal = pyqtSignal(str)

    def __init__(self):
        super(UserDataRecordUI, self).__init__()
        loadUi('./ui/DataRecord.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))

        self.logQueue = queue.Queue()
        self.receiveLogSignal.connect(lambda log: self.popLog(log))
        self.logOutputThread = threading.Thread(target=self.pushLog, daemon=True)
        self.logOutputThread.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)

        self.cap = cv2.VideoCapture()
        self.isExternalCameraUsed = False
        self.useExternalCameraCheckBox.stateChanged.connect(
            lambda: self.useExternalCamera(self.useExternalCameraCheckBox))
        self.startWebcamButton.toggled.connect(self.startWebcam)
        self.startWebcamButton.setCheckable(True)

        self.faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
        self.isFaceDetectEnabled = False
        self.enableFaceDetectButton.toggled.connect(self.enableFaceDetect)
        self.enableFaceDetectButton.setCheckable(True)

        self.database = './FaceBase.db'
        self.datasets = './datasets'
        self.isDbReady = False
        self.initDbButtonRecord.setIcon(QIcon('./icons/warning.png'))
        self.initDbButtonRecord.clicked.connect(self.initDb)

        self.isUserInfoReady = False
        self.userInfo = {'stu_id': '', 'cn_name': '', 'en_name': ''}
        self.addOrUpdateUserInfoButton.clicked.connect(self.addOrUpdateUserInfo)
        self.migrateToDbButton.clicked.connect(self.migrateToDb)

        self.startFaceRecordButton.clicked.connect(lambda: self.startFaceRecord(self.startFaceRecordButton))
        # self.startFaceRecordButton.setCheckable(True)
        self.faceRecordCount = 0
        self.minFaceRecordCount = 100
        self.isFaceDataReady = False
        self.isFaceRecordEnabled = False
        self.enableFaceRecordButton.clicked.connect(self.enableFaceRecord)

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        if self.timer.isActive():
            self.timer.stop()
        event.accept()

    def pushLog(self):
        while True:
            data = self.logQueue.get()
            if data:
                self.receiveLogSignal.emit(data)
            else:
                continue

    def popLog(self, log):
        self.logTextEdit.moveCursor(QTextCursor.End)
        time = datetime.now().strftime('[%Y/%m/%d %H:%M:%S]')
        self.logTextEdit.insertPlainText(time + ' ' + log + '\n')
        self.logTextEdit.ensureCursorVisible()

    def useExternalCamera(self, useExternalCameraCheckBox):
        if useExternalCameraCheckBox.isChecked():
            self.isExternalCameraUsed = True
        else:
            self.isExternalCameraUsed = False

    def startWebcam(self, status):
        if status:
            if self.isExternalCameraUsed:
                camID = 1
            else:
                camID = 0
            self.cap.open(camID)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = self.cap.read()

            if not ret:
                logging.error('无法调用电脑摄像头{}'.format(camID))
                self.logQueue.put('Error：初始化摄像头失败')
                self.cap.release()
                self.startWebcamButton.setIcon(QIcon('./icons/error.png'))
                self.startWebcamButton.setChecked(False)
            else:
                self.startWebcamButton.setText('关闭摄像头')
                self.enableFaceDetectButton.setEnabled(True)
                self.timer.start(5)
                self.startWebcamButton.setIcon(QIcon('./icons/success.png'))
        else:
            if self.cap.isOpened():
                if self.timer.isActive():
                    self.timer.stop()
                self.cap.release()
                self.faceDetectCaptureLabel.clear()
                self.faceDetectCaptureLabel.setText('<font color=red>摄像头未开启</font>')
                self.startWebcamButton.setText('打开摄像头')
                self.enableFaceDetectButton.setEnabled(False)
                self.startWebcamButton.setIcon(QIcon())

    def startFaceRecord(self, startFaceRecordButton):
        if startFaceRecordButton.text() == '学生人脸采集':
            if self.isFaceDetectEnabled:
                if self.isUserInfoReady:
                    self.addOrUpdateUserInfoButton.setEnabled(False)
                    if not self.enableFaceRecordButton.isEnabled():
                        self.enableFaceRecordButton.setEnabled(True)
                    self.enableFaceRecordButton.setIcon(QIcon())
                    self.startFaceRecordButton.setIcon(QIcon('./icons/success.png'))
                    self.startFaceRecordButton.setText('结束当前人脸采集')
                else:
                    self.startFaceRecordButton.setIcon(QIcon('./icons/error.png'))
                    self.startFaceRecordButton.setChecked(False)
                    self.logQueue.put('Error：操作失败，系统未检测到有效的用户信息')
            else:
                self.startFaceRecordButton.setIcon(QIcon('./icons/error.png'))
                self.logQueue.put('Error：操作失败，请开启人脸检测')
        else:
            if self.faceRecordCount < self.minFaceRecordCount:
                text = '系统当前采集了 <font color=blue>{}</font> 帧图像，采集数据过少会导致较大的识别误差。'.format(self.faceRecordCount)
                informativeText = '<b>请至少采集 <font color=red>{}</font> 帧图像。</b>'.format(self.minFaceRecordCount)
                UserDataRecordUI.createDialog(QMessageBox.Information, text, informativeText, QMessageBox.Ok)

            else:
                text = '系统当前采集了 <font color=blue>{}</font> 帧图像，继续采集可以提高识别准确率。'.format(self.faceRecordCount)
                informativeText = '<b>你确定结束当前人脸采集吗？</b>'
                ret = UserDataRecordUI.createDialog(QMessageBox.Question, text, informativeText,
                                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

                if ret == QMessageBox.Yes:
                    self.isFaceDataReady = True
                    if self.isFaceRecordEnabled:
                        self.isFaceRecordEnabled = False
                    self.enableFaceRecordButton.setEnabled(False)
                    self.enableFaceRecordButton.setIcon(QIcon())
                    self.startFaceRecordButton.setText('学生人脸采集')
                    self.startFaceRecordButton.setEnabled(False)
                    self.startFaceRecordButton.setIcon(QIcon())
                    self.migrateToDbButton.setEnabled(True)

    def updateFrame(self):
        ret, frame = self.cap.read()
        if ret:
            self.displayImage(frame)
            if self.isFaceDetectEnabled:
                detected_frame = self.detectFace(frame)
                self.displayImage(detected_frame)
            else:
                self.displayImage(frame)

    def enableFaceDetect(self, status):
        if self.cap.isOpened():
            if status:
                self.enableFaceDetectButton.setText('关闭人脸检测')
                self.isFaceDetectEnabled = True
            else:
                self.enableFaceDetectButton.setText('开启人脸检测')
                self.isFaceDetectEnabled = False

    def detectFace(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90))

        stu_id = self.userInfo.get('stu_id')

        for (x, y, w, h) in faces:
            if self.isFaceRecordEnabled:
                try:
                    if not os.path.exists('{}/stu_{}'.format(self.datasets, stu_id)):
                        os.makedirs('{}/stu_{}'.format(self.datasets, stu_id))
                    if len(faces) > 1:
                        raise RecordRepeatedError

                    cv2.imwrite('{}/stu_{}/img.{}.jpg'.format(self.datasets, stu_id, self.faceRecordCount + 1),
                                gray[y - 20:y + h + 20, x - 20:x + w + 20])
                except RecordRepeatedError:
                    self.isFaceRecordEnabled = False
                    logging.error('检测到多张人脸或环境干扰')
                    self.logQueue.put('Warning：检测到多张人脸或环境干扰，请解决问题后继续')
                    self.enableFaceRecordButton.setIcon(QIcon('./icons/warning.png'))
                    continue
                except Exception as e:
                    logging.error('写入人脸图像文件到计算机过程中发生异常')
                    self.enableFaceRecordButton.setIcon(QIcon('./icons/error.png'))
                    self.logQueue.put('Error：无法保存人脸图像，采集当前捕获帧失败')
                else:
                    self.enableFaceRecordButton.setIcon(QIcon('./icons/success.png'))
                    self.faceRecordCount = self.faceRecordCount + 1
                    self.isFaceRecordEnabled = False
                    self.faceRecordCountLcdNum.display(self.faceRecordCount)
            cv2.rectangle(frame, (x - 5, y - 10), (x + w + 5, y + h + 10), (0, 0, 255), 2)

        return frame

    def displayImage(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        self.faceDetectCaptureLabel.setPixmap(QPixmap.fromImage(outImage))
        self.faceDetectCaptureLabel.setScaledContents(True)

    def enableFaceRecord(self):
        if not self.isFaceRecordEnabled:
            self.isFaceRecordEnabled = True

    def initDb(self):
        conn = sqlite3.connect(self.database)
        cursor = conn.cursor()
        try:
            if not os.path.isdir(self.datasets):
                os.makedirs(self.datasets)

            cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                              stu_id VARCHAR(12) PRIMARY KEY NOT NULL,
                              face_id INTEGER DEFAULT -1,
                              cn_name VARCHAR(10) NOT NULL,
                              en_name VARCHAR(16) NOT NULL,
                              created_time DATE DEFAULT (date('now','localtime'))
                              )
                          ''')
            cursor.execute('SELECT Count(*) FROM users')
            result = cursor.fetchone()
            dbUserCount = result[0]
        except Exception as e:
            logging.error('读取数据库异常，无法完成数据库初始化')
            self.isDbReady = False
            self.initDbButtonRecord.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：初始化数据库失败')
        else:
            self.isDbReady = True
            self.dbUserCountLcdNum.display(dbUserCount)
            self.logQueue.put('Success：数据库初始化完成')
            self.initDbButtonRecord.setIcon(QIcon('./icons/success.png'))
            self.initDbButtonRecord.setEnabled(False)
            self.addOrUpdateUserInfoButton.setEnabled(True)
        finally:
            cursor.close()
            conn.commit()
            conn.close()

    def addOrUpdateUserInfo(self):
        self.userInfoDialog = InsertUserInfo()

        id, chineseName, englishName = self.userInfo.get('stu_id'), self.userInfo.get('cn_name'), self.userInfo.get(
            'en_name')
        self.userInfoDialog.stuIDLineEdit.setText(id)
        self.userInfoDialog.cnNameLineEdit.setText(chineseName)
        self.userInfoDialog.enNameLineEdit.setText(englishName)

        self.userInfoDialog.okButton.clicked.connect(self.checkToApplyUserInfo)
        self.userInfoDialog.exec()

    def checkToApplyUserInfo(self):
        if not (self.userInfoDialog.stuIDLineEdit.hasAcceptableInput() and
                self.userInfoDialog.cnNameLineEdit.hasAcceptableInput() and
                self.userInfoDialog.enNameLineEdit.hasAcceptableInput()):
            self.userInfoDialog.msgLabel.setText('<font color=red>你的输入有误，提交失败，请检查并重试！</font>')
        else:
            self.userInfo['stu_id'] = self.userInfoDialog.stuIDLineEdit.text().strip()
            self.userInfo['cn_name'] = self.userInfoDialog.cnNameLineEdit.text().strip()
            self.userInfo['en_name'] = self.userInfoDialog.enNameLineEdit.text().strip()

            stu_id, cn_name, en_name = self.userInfo.get('stu_id'), self.userInfo.get('cn_name'), self.userInfo.get(
                'en_name')
            self.stuIDLineEdit.setText(stu_id)
            self.cnNameLineEdit.setText(cn_name)
            self.enNameLineEdit.setText(en_name)

            self.isUserInfoReady = True
            if not self.startFaceRecordButton.isEnabled():
                self.startFaceRecordButton.setEnabled(True)
            self.migrateToDbButton.setIcon(QIcon())

            self.userInfoDialog.close()

    def migrateToDb(self):
        if self.isFaceDataReady:
            stu_id, cn_name, en_name = self.userInfo.get('stu_id'), self.userInfo.get('cn_name'), self.userInfo.get(
                'en_name')
            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()

            try:
                cursor.execute('SELECT * FROM users WHERE stu_id=?', (stu_id,))
                if cursor.fetchall():
                    text = '数据库已存在学号为 <font color=blue>{}</font> 的用户记录。'.format(stu_id)
                    informativeText = '<b>是否覆盖？</b>'
                    ret = UserDataRecordUI.createDialog(QMessageBox.Warning, text, informativeText,
                                                        QMessageBox.Yes | QMessageBox.No)

                    if ret == QMessageBox.Yes:
                        cursor.execute('UPDATE users SET cn_name=?, en_name=? WHERE stu_id=?',
                                       (cn_name, en_name, stu_id,))
                    else:
                        raise OperationBeInteruptedError
                else:
                    cursor.execute('INSERT INTO users (stu_id, cn_name, en_name) VALUES (?, ?, ?)',
                                   (stu_id, cn_name, en_name,))

                cursor.execute('SELECT Count(*) FROM users')
                result = cursor.fetchone()
                dbUserCount = result[0]
            except OperationBeInteruptedError:
                pass
            except Exception as e:
                logging.error('读写数据库异常，无法向数据库插入/更新记录')
                self.migrateToDbButton.setIcon(QIcon('./icons/error.png'))
                self.logQueue.put('Error：读写数据库异常，同步失败')
            else:
                text = '<font color=blue>{}</font> 已添加/更新到数据库。'.format(stu_id)
                informativeText = '<b><font color=blue>{}</font> 的人脸数据采集已完成！</b>'.format(cn_name)
                UserDataRecordUI.createDialog(QMessageBox.Information, text, informativeText, QMessageBox.Ok)

                for key in self.userInfo.keys():
                    self.userInfo[key] = ''
                self.isUserInfoReady = False

                self.faceRecordCount = 0
                self.isFaceDataReady = False
                self.faceRecordCountLcdNum.display(self.faceRecordCount)
                self.dbUserCountLcdNum.display(dbUserCount)

                self.stuIDLineEdit.clear()
                self.cnNameLineEdit.clear()
                self.enNameLineEdit.clear()
                self.migrateToDbButton.setIcon(QIcon('./icons/success.png'))

                self.addOrUpdateUserInfoButton.setEnabled(True)
                self.migrateToDbButton.setEnabled(False)

            finally:
                cursor.close()
                conn.commit()
                conn.close()
        else:
            self.logQueue.put('Error：操作失败，你尚未完成人脸数据采集')
            self.migrateToDbButton.setIcon(QIcon('./icons/error.png'))

    @staticmethod
    def createDialog(icon, text, informativeText, standardButtons, defaultButton=None):
        msg = QMessageBox()
        msg.setWindowIcon(QIcon('./icons/icon.png'))
        msg.setWindowTitle('UserRecord System')
        msg.setIcon(icon)
        msg.setText(text)
        msg.setInformativeText(informativeText)
        msg.setStandardButtons(standardButtons)
        if defaultButton:
            msg.setDefaultButton(defaultButton)
        return msg.exec()

class InsertUserInfo(QDialog):
    def __init__(self):
        super(InsertUserInfo, self).__init__()
        loadUi('./ui/UserInfoDialog.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setFixedSize(550, 400)
        self.setStyleSheet(open(os.path.join('all.qss')).read())
        #self.setWindowFlag(QtCore.Qt.FramelessWindowHint)

        idValid = QRegExpValidator(QRegExp('^[0-9]{8}$'), self.stuIDLineEdit)
        self.stuIDLineEdit.setValidator(idValid)
        chineseValid = QRegExpValidator(QRegExp('^[\u4e00-\u9fa5]{1,5}$'), self.cnNameLineEdit)
        self.cnNameLineEdit.setValidator(chineseValid)
        englishValid = QRegExpValidator(QRegExp('^[ A-Za-z]{1,16}$'), self.enNameLineEdit)
        self.enNameLineEdit.setValidator(englishValid)

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        QPainter(self).drawPixmap(self.rect(), QPixmap("./背景图.png"))

class OperationBeInteruptedError(Exception):
    pass

class RecordRepeatedError(Exception):
    pass

if __name__ == '__main__':
    logging.config.fileConfig('./config/logging.cfg')
    app = QApplication(sys.argv)
    window = UserDataRecordUI()
    window.show()
    sys.exit(app.exec())
