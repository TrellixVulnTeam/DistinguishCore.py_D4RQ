from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon, QTextCursor
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
import logging
import logging.config
import sqlite3
import sys
import threading
import queue
import multiprocessing
from datetime import datetime
import dlib
import datetime
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import os
import cv2
import winsound
from tensorflow.python.keras.models import load_model
from keras.preprocessing import image
import time
from playsound import playsound

model = load_model('model.h5')


class UserRecordWindow(QWidget):
    def __init__(self):
        super(UserRecordWindow, self).__init__()
        loadUi('./ui/DataRecord.ui', self)

    def open(self):
        self.show()


class MainUI(QMainWindow):
    logQueue = multiprocessing.Queue()
    receiveLogSignal = pyqtSignal(str)
    databaseFile = './FaceBase.db'
    trainingDataFile = './recognizer/trainingData.yml'
    cap = cv2.VideoCapture()
    captureQueue = queue.Queue()

    def __init__(self):
        super(MainUI, self).__init__()
        loadUi('./ui/Core.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.isExternalCameraUsed = False
        self.useExternalCameraCheckBox.stateChanged.connect(
            lambda: self.useExternalCamera(self.useExternalCameraCheckBox))
        self.faceProcessingThread = UserDistinguishProgramThread()
        self.startWebcamButton.clicked.connect(self.startWebcam)
        self.initDbButton.clicked.connect(self.initDb)
        self.mp3Button.stateChanged.connect(
            lambda: self.faceProcessingThread.useMp3(self.mp3Button))



        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)

        self.faceTrackerCheckBox.stateChanged.connect(
            lambda: self.faceProcessingThread.enableFaceTracker(self))
        self.faceRecognizerCheckBox.stateChanged.connect(
            lambda: self.faceProcessingThread.enableFaceRecognizer(self))
        self.panalarmCheckBox.stateChanged.connect(lambda: self.faceProcessingThread.enablePanalarm(self))

        self.equalizeHistCheckBox.stateChanged.connect(
            lambda: self.faceProcessingThread.enableEqualizeHist(self))

        self.debugCheckBox.stateChanged.connect(lambda: self.faceProcessingThread.enableDebug(self))
        self.confidenceThresholdSlider.valueChanged.connect(
            lambda: self.faceProcessingThread.setConfidenceThreshold(self))
        self.confidenceThresholdSlider.valueChanged.connect(
            lambda: self.confidenceThresholdWidget.setPowerLevel(self.confidenceThresholdSlider.value()))
        self.autoAlarmThresholdSlider.valueChanged.connect(
            lambda: self.faceProcessingThread.setAutoAlarmThreshold(self))
        self.autoAlarmThresholdSlider.valueChanged.connect(
            lambda: self.autoAlarmThresholdWidget.setPowerLevel(self.autoAlarmThresholdSlider.value()))

        self.alarmSignalThreshold = 10
        self.isBellEnabled = True
        self.isTelegramBotPushEnabled = False

        self.receiveLogSignal.connect(lambda log: self.popLog(log))
        self.logOutputThread = threading.Thread(target=self.pushLog, daemon=True)
        self.logOutputThread.start()

    def pushLog(self):
        while True:
            data = self.logQueue.get()
            if data:
                self.receiveLogSignal.emit(data)
            else:
                continue

    def popLog(self, log):
        self.logTextEdit.moveCursor(QTextCursor.End)
        time = datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]')
        self.logTextEdit.insertPlainText(time + ' ' + log + '\n')
        self.logTextEdit.ensureCursorVisible()

    def initDb(self):
        try:
            if not os.path.isfile(self.databaseFile):
                raise DatabaseNotFound
            if not os.path.isfile(self.trainingDataFile):
                raise TrainingDataNotFound

            conn = sqlite3.connect(self.databaseFile)
            cursor = conn.cursor()
            cursor.execute('SELECT Count(*) FROM users')
            result = cursor.fetchone()
            dbUserCount = result[0]
        except DatabaseNotFound:
            logging.error('系统找不到数据库文件{}'.format(self.databaseFile))
            # self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：未发现数据库文件，你可能未进行人脸采集')
        except TrainingDataNotFound:
            logging.error('系统找不到已训练的人脸数据{}'.format(self.trainingDataFile))
            # self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：未发现已训练的人脸数据文件，请完成训练后继续')
        except Exception as e:
            logging.error('读取数据库异常，无法完成数据库初始化')
            # self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：读取数据库异常，初始化数据库失败')
        else:
            cursor.close()
            conn.close()
            if not dbUserCount > 0:
                logging.warning('数据库为空')
                self.logQueue.put('warning：数据库为空，人脸识别功能不可用')
                # self.initDbButton.setIcon(QIcon('./icons/warning.png'))
            else:
                self.logQueue.put('数据库状态正常，发现用户数：{}'.format(dbUserCount))
                self.initDbButton.setText("已链接数据库")
                self.initDbButton.setIcon(QIcon('./icons/success.png'))
                #self.initDbButton.setEnabled(False)
                self.faceRecognizerCheckBox.setToolTip('须先开启注意力检测')
                self.faceRecognizerCheckBox.setEnabled(True)

    def useExternalCamera(self, useExternalCameraCheckBox):
        if useExternalCameraCheckBox.isChecked():
            self.isExternalCameraUsed = True
        else:
            self.isExternalCameraUsed = False

    def startWebcam(self):
        if not self.cap.isOpened():
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
            else:
                self.faceProcessingThread.start()
                self.timer.start(5)
                self.startWebcamButton.setIcon(QIcon('./icons/success.png'))
                self.startWebcamButton.setText('关闭摄像头')

        else:
            text = '如果关闭摄像头，须重启程序才能再次打开。'
            informativeText = '<b>是否继续？</b>'
            ret = MainUI.callDialog(QMessageBox.Warning, text, informativeText, QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)

            if ret == QMessageBox.Yes:
                self.faceProcessingThread.stop()
                if self.cap.isOpened():
                    if self.timer.isActive():
                        self.timer.stop()
                    self.cap.release()

                self.realTimeCaptureLabel.clear()
                self.realTimeCaptureLabel.setText('<font color=red>摄像头未开启</font>')
                self.startWebcamButton.setText('摄像头已关闭')
                self.startWebcamButton.setEnabled(False)
                self.startWebcamButton.setIcon(QIcon())

    def updateFrame(self):
        if self.cap.isOpened():
            if not self.captureQueue.empty():
                captureData = self.captureQueue.get()
                realTimeFrame = captureData.get('realTimeFrame')
                self.displayImage(realTimeFrame, self.realTimeCaptureLabel)

    def displayImage(self, img, qlabel):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        qlabel.setPixmap(QPixmap.fromImage(outImage))
        qlabel.setScaledContents(True)

    def callDialog(icon, text, informativeText, standardButtons, defaultButton = None):
        msg = QMessageBox()
        msg.setWindowIcon(QIcon('./icons/icon.png'))
        msg.setWindowTitle('OpenCV Face Recognition System - Core')
        msg.setIcon(icon)
        msg.setText(text)
        msg.setInformativeText(informativeText)
        msg.setStandardButtons(standardButtons)
        if defaultButton:
            msg.setDefaultButton(defaultButton)
        return msg.exec()

    def closeEvent(self, event):
        if self.faceProcessingThread.isRunning:
            self.faceProcessingThread.stop()
        if self.timer.isActive():
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


class UserDistinguishProgramThread(QThread):
    logQueue1 = MainUI.logQueue

    def __init__(self):
        super(UserDistinguishProgramThread, self).__init__()
        self.detector = dlib.get_frontal_face_detector()
        print("加载相关组件ing")
        self.predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

        self.faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
        print("加载完毕")
        self.playmp3button = {"nod.mp3": False, "shakeHead.mp3": False, "blink.mp3": False, "alarm.wav": False}
        self.playmp3 = True
        self.AR_CONSEC_FRAMES_check = 3
        self.OUT_AR_CONSEC_FRAMES_check = 5
        self.EYE_AR_THRESH = 0.15
        self.EYE_AR_CONSEC_FRAMES = self.AR_CONSEC_FRAMES_check

        self.MAR_THRESH = 0.8
        self.MOUTH_AR_CONSEC_FRAMES = self.AR_CONSEC_FRAMES_check

        self.HAR_THRESH = 15
        self.NOD_AR_CONSEC_FRAMES = self.AR_CONSEC_FRAMES_check

        self.COUNTER = 0
        self.TOTAL = 0
        self.mCOUNTER = 0
        self.mTOTAL = 0
        self.shake = 0
        self.totalface=0
        self.hCOUNTER = 0
        self.hTOTAL = 0
        self.oCOUNTER = 0
        self.i = 1
        self.text1 = None

        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                                      [1.330353, 7.122144, 6.903745],
                                      [-1.330353, 7.122144, 6.903745],
                                      [-6.825897, 6.760612, 4.402142],
                                      [5.311432, 5.485328, 3.987654],
                                      [1.789930, 5.393625, 4.413414],
                                      [-1.789930, 5.393625, 4.413414],
                                      [-5.311432, 5.485328, 3.987654],
                                      [2.005628, 1.409845, 6.165652],
                                      [-2.005628, 1.409845, 6.165652],
                                      [2.774015, -2.080775, 5.048531],
                                      [-2.774015, -2.080775, 5.048531],
                                      [0.000000, -3.116408, 6.097667],
                                      [0.000000, -7.415691, 4.070434]])

        self.K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
                  0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
                  0.0, 0.0, 1.0]

        self.D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

        self.TOTAL_SHAKE_HEAD = 0

        self.cam_matrix = np.array(self.K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(self.D).reshape(5, 1).astype(np.float32)

        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                        [10.0, 10.0, -10.0],
                                        [10.0, -10.0, -10.0],
                                        [10.0, -10.0, 10.0],
                                        [-10.0, 10.0, 10.0],
                                        [-10.0, 10.0, -10.0],
                                        [-10.0, -10.0, -10.0],
                                        [-10.0, -10.0, 10.0]])

        self.line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                           [4, 5], [5, 6], [6, 7], [7, 4],
                           [0, 4], [1, 5], [2, 6], [3, 7]]

        self.isRunning = True

        self.isFaceTrackerEnabled = True
        self.isFaceRecognizerEnabled = False
        self.isPanalarmEnabled = True

        self.isDebugMode = False
        self.confidenceThreshold = 50
        self.autoAlarmThreshold = 65

        self.isEqualizeHistEnabled = False

        self.VIDEO_STREAM = 0
        self.CAMERA_STYLE = False

    def enableFaceTracker(self, coreUI):
        if coreUI.faceTrackerCheckBox.isChecked():
            self.isFaceTrackerEnabled = True
            coreUI.statusBar().showMessage('注意力检测：开启')
        else:
            self.isFaceTrackerEnabled = False
            coreUI.statusBar().showMessage('注意力检测：关闭')

    def enableFaceRecognizer(self, coreUI):
        if coreUI.faceRecognizerCheckBox.isChecked():
            if self.isFaceTrackerEnabled:
                self.isFaceRecognizerEnabled = True
                coreUI.statusBar().showMessage('人脸识别：开启')
            else:
                MainUI.logQueue.put('Error：操作失败，请先开启注意力检测')
                coreUI.faceRecognizerCheckBox.setCheckState(Qt.Unchecked)
                coreUI.faceRecognizerCheckBox.setChecked(False)
        else:
            self.isFaceRecognizerEnabled = False
            coreUI.statusBar().showMessage('人脸识别：关闭')

    def enablePanalarm(self, coreUI):
        if coreUI.panalarmCheckBox.isChecked():
            self.isPanalarmEnabled = True
            coreUI.statusBar().showMessage('注意力报警 ：开启')
        else:
            self.isPanalarmEnabled = False
            coreUI.statusBar().showMessage('注意力报警：关闭')

    def enableDebug(self, coreUI):
        if coreUI.debugCheckBox.isChecked():
            self.isDebugMode = True
            coreUI.statusBar().showMessage('调试模式：开启')
        else:
            self.isDebugMode = False
            coreUI.statusBar().showMessage('调试模式：关闭')

    def setConfidenceThreshold(self, coreUI):
        if self.isDebugMode:
            self.confidenceThreshold = coreUI.confidenceThresholdSlider.value()
            coreUI.statusBar().showMessage('置信度阈值：{}'.format(self.confidenceThreshold))

    def setAutoAlarmThreshold(self, coreUI):
        if self.isDebugMode:
            self.autoAlarmThreshold = coreUI.autoAlarmThresholdSlider.value()
            coreUI.statusBar().showMessage('自动报警阈值：{}'.format(self.autoAlarmThreshold))

    def enableEqualizeHist(self, coreUI):
        if coreUI.equalizeHistCheckBox.isChecked():
            self.isEqualizeHistEnabled = True
            coreUI.statusBar().showMessage('直方图均衡化：开启')
        else:
            self.isEqualizeHistEnabled = False
            coreUI.statusBar().showMessage('直方图均衡化：关闭')

    def yawn(self, shape, mStart, mEnd):
        mouth = shape[mStart:mEnd]
        mar = self.mouthRatio(mouth)
        if mar > self.MAR_THRESH:
            self.mCOUNTER += 1
        else:
            if self.mCOUNTER >= self.MOUTH_AR_CONSEC_FRAMES:
                self.mTOTAL += 1
                self.logQueue1.put('您已经打哈欠{}次'.format(self.mTOTAL))
            self.mCOUNTER = 0

    def nod(self, shape, realTimeFrame):
        try:
            reprojectdst, euler_angle = self.getPose(shape)
            har = euler_angle[0, 0]
            end2 = time.time()
            start2 = 0
            if har < 13:
                start2 = time.time()

            if (end2 - start2) > 3.0:
                cv2.putText(realTimeFrame, "TIRED", (200, 325), cv2.FONT_HERSHEY_PLAIN, 7, (0, 0, 255))
                #self.logQueue1.put('同学，已检测出您有玩手机的迹象，请离开手机，专心学习！！！  \n')
                #threading.Thread(target=lambda: winsound.Beep(1000, 1000)).start()
                threading.Thread(target=lambda: self.playMp3("nod.mp3", "同学，已检测出您有玩手机的迹象，请离开手机，专心学习！！！  \n")).start()
                self.TOTAL = 0
                self.mTOTAL = 0
                self.hTOTAL = -1
                self.shake = 0
        except Exception as e:
            print(e)

        if har > self.HAR_THRESH:
            self.hCOUNTER += 1
        else:
            if self.hCOUNTER >= self.NOD_AR_CONSEC_FRAMES:
                self.hTOTAL += 1
                # self.logQueue1.put('您已经瞌睡点头{}次'.format(self.hTOTAL))
            self.hCOUNTER = 0
        return reprojectdst, realTimeFrame

    def blink(self, leftEye, rightEye, realTimeFrame):
        global start1
        leftEAR = self.eyeRatio(leftEye)
        rightEAR = self.eyeRatio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(realTimeFrame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(realTimeFrame, [rightEyeHull], -1, (0, 255, 0), 1)
        end = time.time()
        if ear > 0.16:
            start1 = time.time()

        if (end - start1) > 1.5:
            cv2.putText(realTimeFrame, "TIRED", (200, 325), cv2.FONT_HERSHEY_PLAIN, 7, (0, 0, 255))
            #self.logQueue1.put('同学，已检测出您有睡觉的迹象，请注意，请注意！！！  \n')
            #threading.Thread(target=lambda: winsound.Beep(1000, 1000)).start()
            threading.Thread(target=lambda: self.playMp3("blink.mp3", '同学，已检测出您有睡觉的迹象，请注意，请注意！！！  \n')).start()
            self.TOTAL = 0
            self.mTOTAL = 0
            self.hTOTAL = 0
        if ear < self.EYE_AR_THRESH:
            self.COUNTER += 1
        else:
            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                self.TOTAL += 1
                self.logQueue1.put('同学，您已走神{}次，请睁开双眼，认真听讲'.format(self.TOTAL))
            self.COUNTER = 0
        return realTimeFrame

    def shakeHead(self, nose, jaw, distance_left, distance_right, totalface):
        NOSE_JAW_Distance = self.nose_jaw_distance(nose, jaw)
        face_left1 = NOSE_JAW_Distance[0]
        face_right1 = NOSE_JAW_Distance[1]
        face_left2 = NOSE_JAW_Distance[2]
        face_right2 = NOSE_JAW_Distance[3]

        if face_left1 >= face_right1 + 50 and face_left2 >= face_right2 + 50:
            distance_left += 1
        if face_right1 >= face_left1 + 50 and face_right2 >= face_left2 + 50:
            distance_right += 1
        if distance_left != 0 and distance_right != 0:
            self.TOTAL_SHAKE_HEAD += 1
            self.shake += 1
            distance_right = 0
            distance_left = 0

        if self.TOTAL_SHAKE_HEAD != totalface:
            self.logQueue1.put('同学，您已左顾右盼{}次，请集中注意力，不要分心'.format(self.TOTAL_SHAKE_HEAD))
            #threading.Thread(target=lambda: self.playMp3("shakeHead.mp3", '同学，您已左顾右盼{}次，请集中注意力，不要分心'.format(self.TOTAL_SHAKE_HEAD))).start()
            totalface += 1
        return distance_left, distance_right, totalface

    def getFace(self, face_id, cursor, confidence, realTimeFrame, _x, _y):
        isKnown = False
        try:
            cursor.execute("SELECT * FROM users WHERE face_id=?", (face_id,))
            result = cursor.fetchall()
            if result:
                en_name = result[0][3]
            else:
                raise Exception
        except Exception as e:
            logging.error('读取数据库异常，系统无法获取Face ID为{}的身份信息'.format(face_id))
            MainUI.logQueue.put('Error：读取数据库异常，系统无法获取Face ID为{}的身份信息'.format(face_id))
            en_name = ''

        if confidence < self.confidenceThreshold:
            isKnown = True
            cv2.putText(realTimeFrame, en_name, (_x - 5, _y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 97, 255), 2)
        else:
            cv2.putText(realTimeFrame, 'unknown', (_x - 5, _y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return isKnown, realTimeFrame

    def humanParam(self, faces, realTimeFrame, distance_left, distance_right, totalface, mStart, mEnd, lStart, lEnd, rStart, rEnd, nStart, nEnd, jStart, jEnd):
        for k, d in enumerate(faces):
            try:
                shape = self.predictor(realTimeFrame, d)
                if self.isFaceTrackerEnabled is True:
                    for i in range(68):
                        cv2.circle(realTimeFrame, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
                shape = face_utils.shape_to_np(shape)

                """打哈欠"""
                self.yawn(shape, mStart, mEnd)

                """瞌睡点头"""
                reprojectdst, realTimeFrame = self.nod(shape, realTimeFrame)

                """眨眼"""
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                realTimeFrame = self.blink(leftEye, rightEye, realTimeFrame)

                """摇头"""
                nose = shape[nStart:nEnd]
                jaw = shape[jStart:jEnd]
                distance_left, distance_right, totalface = self.shakeHead(nose, jaw, distance_left, distance_right,
                                                                          totalface)

            except Exception as e:
                print(e)
                continue

        return reprojectdst, distance_left, distance_right, totalface, realTimeFrame

    def getPose(self, shape):
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])
        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, self.cam_matrix,
                                            self.dist_coeffs)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

        return reprojectdst, euler_angle

    def eyeRatio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def mouthRatio(self, mouth):
        A = np.linalg.norm(mouth[2] - mouth[9])
        B = np.linalg.norm(mouth[4] - mouth[7])
        C = np.linalg.norm(mouth[0] - mouth[6])
        mar = (A + B) / (2.0 * C)
        return mar

    def nose_jaw_distance(self, nose, jaw):
        face_left1 = dist.euclidean(nose[0], jaw[0])
        face_left2 = dist.euclidean(nose[3], jaw[2])
        face_right1 = dist.euclidean(nose[0], jaw[16])
        face_right2 = dist.euclidean(nose[3], jaw[14])
        face_distance = (face_left1, face_right1, face_left2, face_right2)
        return face_distance

    def run(self):
        global reprojectdst
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
        (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS['jaw']

        distance_left = 0
        distance_right = 0
        totalface = 0
        frameCounter = 0
        currentFaceID = 0

        faceCascade = self.faceCascade
        faceTrackers = {}

        isTrainingDataLoaded = False
        isDbConnected = False

        while self.isRunning:
            try:
                if MainUI.cap.isOpened():
                    ret, frame = MainUI.cap.read()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if self.isEqualizeHistEnabled:
                        gray = cv2.equalizeHist(gray)
                    faces = faceCascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90))

                    if not isTrainingDataLoaded and os.path.isfile(MainUI.trainingDataFile):
                        recognizer = cv2.face.LBPHFaceRecognizer_create()
                        recognizer.read(MainUI.trainingDataFile)
                        isTrainingDataLoaded = True
                    if not isDbConnected and os.path.isfile(MainUI.databaseFile):
                        conn = sqlite3.connect(MainUI.databaseFile)
                        cursor = conn.cursor()
                        isDbConnected = True

                    captureData = {}
                    realTimeFrame = frame.copy()

                    if self.isFaceTrackerEnabled:

                        fidsToDelete = []

                        for fid in faceTrackers.keys():
                            trackingQuality = faceTrackers[fid].update(realTimeFrame)
                            if trackingQuality < 7:
                                fidsToDelete.append(fid)

                        for fid in fidsToDelete:
                            faceTrackers.pop(fid, None)

                        for (_x, _y, _w, _h) in faces:
                            isKnown = False

                            if self.isFaceRecognizerEnabled:
                                cv2.rectangle(realTimeFrame, (_x, _y), (_x + _w, _y + _h), (232, 138, 30), 2)

                                """cv2.putText(realTimeFrame, "Nod: {}".format(self.hTOTAL), (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                                cv2.putText(realTimeFrame, "Blinks: {}".format(self.TOTAL), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                                cv2.putText(realTimeFrame, "Yawning: {}".format(self.mTOTAL), (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)"""
                                cv2.putText(realTimeFrame, "Face: {}".format(len(faces)), (0, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                face_id, confidence = recognizer.predict(gray[_y:_y + _h, _x:_x + _w])
                                logging.debug('face_id：{}，confidence：{}'.format(face_id, confidence))

                                if self.isDebugMode:
                                    MainUI.logQueue.put('Debug -> face_id：{}，confidence：{}'.format(face_id, confidence))

                                isKnown, realTimeFrame = self.getFace(face_id, cursor, confidence, realTimeFrame, _x,
                                                                      _y)

                            frameCounter += 1

                            if frameCounter % 10 == 0:
                                x = int(_x)
                                y = int(_y)
                                w = int(_w)
                                h = int(_h)

                                x_bar = x + 0.5 * w
                                y_bar = y + 0.5 * h

                                matchedFid = None

                                for fid in faceTrackers.keys():
                                    tracked_position = faceTrackers[fid].get_position()
                                    t_x = int(tracked_position.left())
                                    t_y = int(tracked_position.top())
                                    t_w = int(tracked_position.width())
                                    t_h = int(tracked_position.height())

                                    t_x_bar = t_x + 0.5 * t_w
                                    t_y_bar = t_y + 0.5 * t_h

                                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and
                                            (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                                        matchedFid = fid

                                if not isKnown and matchedFid is None:
                                    tracker = dlib.correlation_tracker()
                                    tracker.start_track(realTimeFrame,
                                                        dlib.rectangle(x - 5, y - 10, x + w + 5, y + h + 10))
                                    faceTrackers[currentFaceID] = tracker
                                    currentFaceID += 1

                        for fid in faceTrackers.keys():
                            tracked_position = faceTrackers[fid].get_position()

                            t_x = int(tracked_position.left())
                            t_y = int(tracked_position.top())
                            t_w = int(tracked_position.width())
                            t_h = int(tracked_position.height())

                            for start, end in self.line_pairs:
                                starts = reprojectdst[start]
                                ends = reprojectdst[end]
                                cv2.line(realTimeFrame, (int(starts[0]), int(starts[1])), (int(ends[0]), int(ends[1])),
                                         (0, 0, 255))
                            # cv2.putText(realTimeFrame, "Nod: {}".format(self.hTOTAL), (t_x, t_y + int(t_w/500)), cv2.FONT_HERSHEY_SIMPLEX, t_w/500, (255, 255, 0), 1)
                            cv2.putText(realTimeFrame, "Yawning: {}".format(self.mTOTAL),
                                        (t_x, t_y + int(t_w / 250) + 20), cv2.FONT_HERSHEY_SIMPLEX, t_w / 500,
                                        (255, 255, 0), 1)
                            cv2.putText(realTimeFrame, "ShakeHead: {}".format(self.shake),
                                        (t_x, t_y + int(t_w / 250) + 40), cv2.FONT_HERSHEY_SIMPLEX, t_w / 500,
                                        (255, 255, 0), 1)
                            cv2.putText(realTimeFrame, "Blink: {}".format(self.TOTAL), (t_x, t_y + int(t_w / 250) + 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, t_w / 500, (255, 255, 0), 1)

                    captureData['originFrame'] = frame
                    captureData['realTimeFrame'] = realTimeFrame
                    MainUI.captureQueue.put(captureData)

                    faces = self.detector(gray, 0)

                    if self.isFaceTrackerEnabled:
                        if (len(faces) != 0):
                            reprojectdst, distance_left, distance_right, totalface, realTimeFrame = self.humanParam(
                                faces, realTimeFrame, distance_left, distance_right, totalface, mStart, mEnd, lStart,
                                lEnd, rStart, rEnd, nStart, nEnd, jStart, jEnd)

                        if self.oCOUNTER >= self.OUT_AR_CONSEC_FRAMES_check:
                            self.oCOUNTER = 0

                        if self.TOTAL >= 20 or self.mTOTAL >= 15 or self.hTOTAL >= 10:
                            cv2.imwrite("./new/%d.jpg" % (self.i), frame)
                            img = cv2.imread("./new/%d.jpg" % (self.i))
                            path = "./haarcascades/haarcascade_frontalface_alt.xml"
                            hc = cv2.CascadeClassifier(path)
                            faces = hc.detectMultiScale(img)

                            for face in faces:
                                imgROI = img[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
                                imgROI = cv2.resize(imgROI, (128, 128), interpolation=cv2.INTER_AREA)
                                self.logQueue1.put('已成功保存本帧照片并使用改进的YOLOv3网络模型进行判断！')
                                cv2.imwrite("./new/%d.jpg" % (self.i), imgROI)

                            img = image.load_img("./new/%d.jpg" % (self.i), target_size=(128, 128))
                            x = image.img_to_array(img)
                            x = np.expand_dims(x, axis=0)
                            y = model.predict(x)
                            if (y[0][0]) > 0.5:
                                threading.Thread(target=lambda: winsound.Beep(1000, 1000)).start()
                                cv2.putText(realTimeFrame, "SLEEPING!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                            (0, 0, 255), 2)
                                print(
                                    'tired tired tired tired tired tired tired tired tired tired tired tired tired tired!')
                                self.logQueue1.put('判断结果为疲惫！！请寻找就近的停车点进行休息！\n')
                                self.i = self.i + 1
                                print('%d' % (self.i))

                                self.TOTAL = 0
                                self.mTOTAL = 0
                                self.hTOTAL = 0
                            continue
                    else:
                        continue
            except Exception as e:
                print(e)
        print("end")

    def stop(self):
        self.isRunning = False
        self.quit()
        self.wait()

    def useMp3(self, mp3Button):
        if mp3Button.isChecked():
            self.playmp3 = True
        else:
            self.playmp3 = False

    def playMp3(self, name, log):
        if self.playmp3:
            if self.playmp3button[name] == False:
                self.playmp3button[name] = True
                self.logQueue1.put(log)
                playsound(name)
                self.playmp3button[name] = False


class TrainingDataNotFound(FileNotFoundError):
    pass


class DatabaseNotFound(FileNotFoundError):
    pass


if __name__ == '__main__':
    logging.config.fileConfig('./config/logging.cfg')
    app = QApplication(sys.argv)
    window = MainUI()
    window.show()
    sys.exit(app.exec())

