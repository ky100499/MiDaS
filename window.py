from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
import threading
import sys

from playsound import playsound

class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

        self.setWindowTitle("Depth Estimation")

        self.show()

        self.__running = True
        th = threading.Thread(target=self.run)
        th.start()

    def initUI(self):
        mainWidget = QtWidgets.QWidget()

        vbox = QtWidgets.QVBoxLayout()
        hbox = QtWidgets.QHBoxLayout()

        radioBox = QtWidgets.QVBoxLayout()
        camIndex = 0
        while True:
            print(camIndex)
            cam = cv2.VideoCapture(camIndex)

            if cam.isOpened():
                btn = QtWidgets.QRadioButton("Camera #%d" % camIndex)
                btn.clicked.connect(self.selectCam(camIndex))
                if camIndex == 0:
                    self.cap = cam
                    btn.setChecked(True)
                radioBox.addWidget(btn)
                camIndex += 1
            else:
                break

        radioBox.addStretch()

        spctWrap = QtWidgets.QHBoxLayout()

        imgLabel = QtWidgets.QLabel()
        img = cv2.imread('./assets/img/Spectrum.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (100, 500), interpolation=cv2.INTER_LINEAR)
        h, w, c = img.shape
        qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        imgLabel.setPixmap(pixmap)

        spctExplWrap = QtWidgets.QVBoxLayout()

        farLabel = QtWidgets.QLabel()
        nearLabel = QtWidgets.QLabel()

        farLabel.setText("Far")
        nearLabel.setText("Near")

        spctExplWrap.addWidget(farLabel)
        spctExplWrap.addStretch()
        spctExplWrap.addWidget(nearLabel)

        spctWrap.addLayout(spctExplWrap)

        spctWrap.addWidget(imgLabel)

        radioBox.addLayout(spctWrap)

        self.label = QtWidgets.QLabel()

        hbox.addWidget(self.label)
        hbox.addLayout(radioBox)

        btn_start = QtWidgets.QPushButton("Camera On")
        btn_stop = QtWidgets.QPushButton("Camera Off")

        vbox.addLayout(hbox)
        vbox.addWidget(btn_start)
        vbox.addWidget(btn_stop)

        mainWidget.setLayout(vbox)
        self.setCentralWidget(mainWidget)

        self.label.resize(1280, 720)

        btn_start.clicked.connect(self.start)
        btn_stop.clicked.connect(self.stop)

    def run(self):
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.label.resize(w, h)

        while self.__running:
            ret, frame = self.cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
                h, w, c = frame.shape
                # print(w, h)
                qImg = QtGui.QImage(frame.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                self.label.setPixmap(pixmap)
            else:
                QtWidgets.QMessageBox.about(self, "Error", "Cannot read frame.")
                break

        self.cap.release()

    def beep(self):
        while self.beepPlaying:
            playsound('./assets/audio/beep.mp3')

    def beepOn(self):
        self.beepPlaying = True
        self.beepThread = threading.Thread(target=self.beep)
        self.beepThread.daemon = True
        self.beepThread.start()

    def beepOff(self):
        self.beepPlaying = False

    def stop(self):
        self.beepOff()
        self.__running = False
        print("stopped..")

    def start(self):
        self.beepOn()
        print("started..")

    def onExit(self):
        print("exit")
        self.stop()

    def selectCam(self, i):
        def sc(_):
            self.cap = cv2.VideoCapture(i)
        return sc


app = QtWidgets.QApplication([])
window = Window()
app.aboutToQuit.connect(window.onExit)

sys.exit(app.exec_())
