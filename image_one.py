# tensorflow 2+
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QLabel, QHBoxLayout, QVBoxLayout

from PyQt5.QtGui import *
from PyQt5.QtCore import QCoreApplication


facenet = cv2.dnn.readNet('models/res10_300x300_ssd_iter_140000.caffemodel','models/deploy.prototxt' )
model = load_model('models/mask_detector.model')
# PyQt5 를 이용한 코드 ------------------------------------
class MyApp(QWidget):

    vbox = QVBoxLayout() #수직박스레이어

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('mask_dectecter_안전제일_0419')
        self.setWindowIcon(QIcon('moja.jpg'))
      
        hbox = QHBoxLayout() #수평박스레이어
        hbox.addStretch(1)

        btn_Picture = QPushButton('Picture',self)
        btn_Picture.clicked.connect(self.btn_Picture_clicked) #버튼 클릭시 실행되는 function
        hbox.addWidget(btn_Picture)
        hbox.addStretch(1)

        btn_Quit = QPushButton('Quit',self)
        btn_Quit.clicked.connect(QCoreApplication.instance().quit) #quit을 누르면 나가기
        hbox.addWidget(btn_Quit)
        hbox.addStretch(1)

        
        self.vbox.addStretch(3) #비율 3만큼 추가
        self.vbox.addLayout(hbox) #수직레이어 안쪽에 수평레이어 넣기
        self.vbox.addStretch(1)

        self.setLayout(self.vbox) #메인 레이아웃은 수직박스!

        self.setGeometry(300, 300, 300, 200)
        self.show()
        

    def btn_Picture_clicked(self): #클릭됨ㄴ 실행된는 메서드

        
        #여기서부터 cv2를 이용한 코드--------------------------------
        cap = cv2.VideoCapture(0)#내 웹캠에 연결
        
        ret, frame = cap.read()    # Read 결과와 frame

        cv2.imwrite('webcam_1.jpg',frame)
        self.picture_1 = QLabel() #레이블 생성
        self.qPixmapFileVar = QPixmap() #QPixmap객체 생성
        self.qPixmapFileVar.load("webcam_1.jpg") #상대경로 읽기
        self.picture_1.setPixmap(self.qPixmapFileVar) #보여주기

        self.vbox.addWidget(self.picture_1)
        #-------------face detect---------------------
        
        img = cv2.imread('webcam_1.jpg')
        h, w = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
        facenet.setInput(blob)
        dets = facenet.forward()

        faces = []

        for i in range(dets.shape[2]):
            confidence = dets[0, 0, i, 2]
            if confidence < 0.5:
                continue

            x1 = int(dets[0, 0, i, 3] * w)
            y1 = int(dets[0, 0, i, 4] * h)
            x2 = int(dets[0, 0, i, 5] * w)
            y2 = int(dets[0, 0, i, 6] * h)

            face = img[y1:y2, x1:x2]
            faces.append(face)

        #---------------------mask detect ---------------------------------------------
        for i, face in enumerate(faces):
            face_input = cv2.resize(face, dsize=(224, 224))
            face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
            face_input = preprocess_input(face_input)
            face_input = np.expand_dims(face_input, axis=0)

            mask, nomask = model.predict(face_input).squeeze()


            self.picture_maksper = QLabel('%.2f%%' % (mask * 100))
            self.vbox.addWidget(self.picture_maksper)
        #---------------------------------------------
        cap.release() # 풀기
        

if __name__ == '__main__':
        app = QApplication(sys.argv)
        ex = MyApp()
        sys.exit(app.exec_())

# cv2 닫기

cv2.destroyAllWindows()