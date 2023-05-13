"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import utils
import cv2
import argparse
import time

import numpy as np

from imutils.video import VideoStream
from midas.model_loader import default_models, load_model

from PyQt5 import QtWidgets, QtGui, QtCore
import threading
import sys

from playsound import playsound

class Window(QtWidgets.QMainWindow):
    def __init__(self, model_type, optimize, side, grayscale):
        super().__init__()

        self.initUI()

        self.setWindowTitle("Depth Estimation")

        self.show()

        self.__model_type = model_type
        self.__optimize = optimize
        self.__side = side
        self.__grayscale = grayscale

        self.__running = True
        self.__estimate = False
        th = threading.Thread(target=self.run)
        th.start()

        self.beepThread = None
        self.__beepPlaying = False
        self.dangerLimit = 108

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

        with torch.no_grad():
            while self.__running:
                ret, frame = self.cap.read()
                if not ret:
                    QtWidgets.QMessageBox.about(self, "Error", "Cannot read frame.")
                    break
                elif not self.__estimate:
                    # No Estimation
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
                    h, w, c = frame.shape
                    # print(w, h)
                    qImg = QtGui.QImage(frame.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(qImg)
                    self.label.setPixmap(pixmap)
                else:
                    original_image_rgb = np.flip(frame, 2)  # in [0, 255] (flip required to get RGB)
                    image = transform({"image": original_image_rgb/255})["image"]

                    prediction = process(device, model, self.__model_type, image, (net_w, net_h),
                                            original_image_rgb.shape[1::-1], self.__optimize, True)

                    original_image_bgr = np.flip(original_image_rgb, 2) if self.__side else None
                    content = create_side_by_side(original_image_bgr, prediction, self.__grayscale)

                    # Canny Edge Detection
                    blurred = cv2.GaussianBlur(content, (7, 7), 0)
                    thresh = cv2.Canny(blurred, 20, 150)

                    # 모든 컨투어를 트리 계층 으로 수집
                    contour2, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # print(len(contour2), hierarchy)

                    dangerous = False

                    # 모든 컨투어 그리기
                    for idx, cont in enumerate(contour2):
                        # 랜덤한 컬러 추출
                        color = [int(i) for i in np.random.randint(0,255, 3)]

                        # convexHull
                        approx = cv2.convexHull(cont)

                        # 컨투어 인덱스 마다 랜덤한 색상으로 그리기
                        mask = np.zeros(frame.shape[:2], dtype="uint8")

                        cv2.drawContours(mask, [approx], 0, 255, cv2.FILLED)
                        cv2.drawContours(frame, [approx], 0, color, 3)

                        mask_inv = cv2.bitwise_not(mask)

                        average = np.array(cv2.mean(content, mask=mask)[:3], dtype=np.uint8)

                        c = np.tile(average, frame.shape[:2]+(1,))

                        avg_gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)[0][0]
                        dangerous = dangerous or avg_gray > self.dangerLimit

                        masked_fg = cv2.bitwise_and(c, c, mask=mask)
                        masked_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

                        frame = cv2.add(masked_bg, masked_fg)

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
                    h, w, c = frame.shape
                    qImg = QtGui.QImage(frame.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(qImg)
                    self.label.setPixmap(pixmap)

                    self.beepOn() if dangerous else self.beepOff()

        self.cap.release()

    def beep(self):
        while self.__estimate and self.__beepPlaying:
            playsound('./assets/audio/beep.mp3')

    def beepOn(self):
        if self.beepThread is None or not self.beepThread.is_alive():
            self.__beepPlaying = True
            self.beepThread = threading.Thread(target=self.beep)
            self.beepThread.daemon = True
            self.beepThread.start()

    def beepOff(self):
        self.__beepPlaying = False

    def stop(self):
        self.__estimate = False
        print("stopped..")

    def start(self):
        self.__estimate = True
        print("started..")

    def onExit(self):
        self.__running = False
        print("exit")

    def selectCam(self, i):
        def sc(_):
            self.cap = cv2.VideoCapture(i)
        return sc

first_execution = True
def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


def create_side_by_side(image, depth, grayscale):
    """
    Take an RGB image and depth map and place them side by side. This includes a proper normalization of the depth map
    for better visibility.

    Args:
        image: the RGB image
        depth: the depth map
        grayscale: use a grayscale colormap?

    Returns:
        the image and depth map place side by side
    """
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3

    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    if image is None:
        return right_side
    else:
        return np.concatenate((image, right_side), axis=1)


def run(input_path, output_path, model_path, model_type="dpt_beit_large_512", optimize=False, side=False, height=None,
        square=False, grayscale=False):
    print("Initialize")

    # select device
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device: %s" % device)

    global model, transform, net_w, net_h
    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)

    print("Start processing")

    app = QtWidgets.QApplication([])
    window = Window(model_type, optimize, side, grayscale)
    app.aboutToQuit.connect(window.onExit)

    sys.exit(app.exec_())

    print("Start GUI")

    print()

    print("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path',
                        default=None,
                        help='Folder with input images (if no input path is specified, images are tried to be grabbed '
                             'from camera)'
                        )

    parser.add_argument('-o', '--output_path',
                        default=None,
                        help='Folder for output images'
                        )

    parser.add_argument('-m', '--model_weights',
                        default=None,
                        help='Path to the trained weights of model'
                        )

    parser.add_argument('-t', '--model_type',
                        default='dpt_swin2_large_384',
                        help='Model type: '
                             'dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, '
                             'dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, '
                             'dpt_levit_224, dpt_large_384, dpt_hybrid_384, midas_v21_384, midas_v21_small_256 or '
                             'openvino_midas_v21_small_256'
                        )

    parser.add_argument('-s', '--side',
                        action='store_true',
                        help='Output images contain RGB and depth images side by side'
                        )

    parser.add_argument('--optimize', dest='optimize', action='store_true', help='Use half-float optimization')
    parser.set_defaults(optimize=False)

    parser.add_argument('--height',
                        type=int, default=None,
                        help='Preferred height of images feed into the encoder during inference. Note that the '
                             'preferred height may differ from the actual height, because an alignment to multiples of '
                             '32 takes place. Many models support only the height chosen during training, which is '
                             'used automatically if this parameter is not set.'
                        )
    parser.add_argument('--square',
                        action='store_true',
                        help='Option to resize images to a square resolution by changing their widths when images are '
                             'fed into the encoder during inference. If this parameter is not set, the aspect ratio of '
                             'images is tried to be preserved if supported by the model.'
                        )
    parser.add_argument('--grayscale',
                        action='store_true',
                        help='Use a grayscale colormap instead of the inferno one. Although the inferno colormap, '
                             'which is used by default, is better for visibility, it does not allow storing 16-bit '
                             'depth values in PNGs but only 8-bit ones due to the precision limitation of this '
                             'colormap.'
                        )

    args = parser.parse_args()


    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(args.input_path, args.output_path, args.model_weights, args.model_type, args.optimize, args.side, args.height,
        args.square, args.grayscale)
