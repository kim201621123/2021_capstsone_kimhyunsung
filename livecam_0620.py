# tensorflow 2+
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QLabel, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import QCoreApplication
import requests
import json
import qrcode


facenet = cv2.dnn.readNet('models/res10_300x300_ssd_iter_140000.caffemodel','models/deploy.prototxt' )
model = load_model('models/mask_detector.model')


key_index = 1  #이걸로 개인키인 code와 kakaojson파일 바꿀것이다.

#   0 car의 키 personal_key = ""
#   1 kim의 키 personal_key = ""
#이게 개인이 로그인페이지 - 로컬호스트에서 회신받는 code


rest_api_key = "" #앱 만들면 부여받는 api키  현재는 mask앱꺼

personal_key = ["wJSrVzCXInVyHOHgW2mQywWhR8YOcwfz_lpRco4uysq37Q5JMiBt4BQUOCe6i6qCpz4BLgorDKgAAAF50ri8Cw", #car  0번
                "IrTDMU5V-CLLNgImO0Xo052mryR8aOJ6O8jXh-4jBU1tI7GtfyZ49fqNOpptlGq3B5FMhwo9dVoAAAF6J6jenA", #kim  1번
                ""
                ]


#   "https://kauth.kakao.com/oauth/authorize?client_id=c98fe6d1bf6316d9d186ec12d3181f6b&response_type=code&scope=talk_message,friends&redirect_uri=https://localhost.com"
#    https://kauth.kakao.com/oauth/authorize?client_id=c98fe6d1bf6316d9d186ec12d3181f6b&redirect_uri=https://localhost.com&response_type=code&scope=talk_message,friends

def make_qr_code():
    kakao_login_url = "https://kauth.kakao.com/oauth/authorize?client_id=" + rest_api_key + "&response_type=code&redirect_uri=https://localhost.com"
    img = qrcode.make(kakao_login_url)
    img.save("C:\\Users\pinoc\mask-detection-master\mask-detection-master1\imgs\img_qr.jpg") #unicode 때문에 u앞에 \\해야함
    
    #print(type(img))
    #print(img.size)

    img_qr = "C:\\Users\pinoc\mask-detection-master\mask-detection-master1\imgs\img_qr.jpg"
    gray = cv2.imread(img_qr, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('qrcode_kakao_login', gray)
    
    cv2.waitKey(1) == ord('q')
           

def kakao_token():
    url = "https://kauth.kakao.com/oauth/token"

    data = {
        "grant_type" : "authorization_code",
        "client_id" : rest_api_key, #내 앱 id
        "redirect_uri" : "https://localhost.com",
        "code" : personal_key[key_index] #개인 key
    }

    response = requests.post(url, data=data)

    tokens = response.json()

    print(tokens)

    with open("kakao_tokens" + str(key_index) + ".json", "w") as fp:  #key_index에 따라서 저장
        json.dump(tokens, fp)
    #----------------------위는 개인 접근 코드를 이용해 토큰을 받아오기 / 밑은 그걸 읽어와서 메시지 보내기


def kakao_token_mesage_to_friends():
   
    #----------------------
    with open("kakao_tokens" + str(key_index) + ".json", "r") as fp: #key_index에 따라서 읽기
        tokens = json.load(fp)
    
    
    friend_url = "https://kapi.kakao.com/v1/api/talk/friends"

    headers={"Authorization" : "Bearer " + tokens["access_token"],
             "limits" : "3"
             }

    result = json.loads(requests.get(friend_url, headers=headers).text)

    print(type(result))
    print("=============================================")
    print(result)
    print("=============================================")
    friends_list = result.get("elements")
    print(friends_list)
    # print(type(friends_list))
    print("=============================================") #친구리스트의 맨 마지막으로 추가된 사람에게 메시지를 보낸다.
    print(friends_list[-1].get("uuid"))
    friend_id = friends_list[-1].get("uuid")
    print(friend_id)

    send_url= "https://kapi.kakao.com/v1/api/talk/friends/message/default/send"

    # 사용자 토큰
    headers = {
        "Authorization": "Bearer " + tokens["access_token"],
        "limits" : "3"
    }

    data={
            'receiver_uuids': '["{}"]'.format(friend_id),
            "template_object": json.dumps({
               "object_type": "feed",
                "content": {
                    "title": "마스크를 미착용하셨습니다.",
                    "description": "어떤 정보를 원하십니까?",
                    "image_url": "", #https://ifh.cc/g/mSy9nv.jpg 현재는 이미지를 서버의 이미지를 이지지 주소로 만든것을 보내는 중
                    "image_width": 115,
                    "image_height": 171,
                    "link": {
                    "web_url": "https://www.naver.com",
                    "mobile_web_url": "https://www.naver.com",
                    "android_execution_params": "contentId=100",
                    "ios_execution_params": "contentId=100"
                    }
                },
                "buttons": [
                    {
                    "title": "백신예약 페이지",
                    "link": {
                        "web_url": "https://ncvr.kdca.go.kr",   #백신예약페이지
                        "mobile_web_url": "https://ncvr.kdca.go.kr"
                    }
                    },
                    {
                    "title": "마스크 파는 곳",
                    "link": {
                        "web_url":"https://www.google.com/search?q=%EC%95%BD%EA%B5%AD",  #기기의 위치에서 가까운 약국 & 편의점 지도
                        "mobile_web_url":"https://www.google.com/search?q=%EC%95%BD%EA%B5%AD",
                        "android_execution_params": "contentId=100",
                        "ios_execution_params": "contentId=100"
                    }
                    }
                ]
            })
        }
    response = requests.post(send_url, headers=headers, data=data)
    print(response.status_code)
    if response.json().get('result_code') == 0:
        print('메시지를 성공적으로 보냈습니다.')
    else:
        print('text : ' + str(response.json()))

#-------------이건 특정 사용자에게 보내는 코드
def kakao_token_mesage_to_someone():
    
    with open("kakao_tokens" + str(key_index) + ".json", "r") as fp:
        tokens = json.load(fp)

    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"

    # 사용자 토큰
    headers = {
        "Authorization": "Bearer " + tokens["access_token"]
    }

    data={
            "template_object": json.dumps({
               "object_type": "feed",
                "content": {
                    "title": " 특정 사용자에게 보내는 메시지",
                    "description": "1",
                    "image_url": "", #보류
                    "image_width": 115,
                    "image_height": 171,
                    "link": {
                    "web_url": "https://www.naver.com",
                    "mobile_web_url": "https://www.naver.com",
                    "android_execution_params": "contentId=100",
                    "ios_execution_params": "contentId=100"
                    }
                },
                "buttons": [
                    {
                    "title": "네",
                    "link": {
                        "web_url": "https://www.naver.com",
                        "mobile_web_url": "https://www.naver.com"
                    }
                    },
                    {
                    "title": "아니오",
                    "link": {
                        "web_url":"https://www.google.com/search?q=%EC%95%BD%EA%B5%AD",
                        "mobile_web_url":"https://www.google.com/search?q=%EC%95%BD%EA%B5%AD",
                        "android_execution_params": "contentId=100",
                        "ios_execution_params": "contentId=100"
                    }
                    }
                ]
            })
        }
    response = requests.post(url, headers=headers, data=data)
    print(response.status_code)
    if response.json().get('result_code') == 0:
        print('메시지를 성공적으로 보냈습니다.')
    else:
        print(' text ' + str(response.json()))



vertical_layout = QtWidgets.QVBoxLayout()
horizontal_layout = QtWidgets.QHBoxLayout()


class ShowVideo(QtCore.QObject):

    flag = 0

    global no_maksper

    camera = cv2.VideoCapture(0)

    ret, image = camera.read()
    height, width = image.shape[:2]

    make_qr_code()      #현재는 mask 앱 키로 url qr만듬

    VideoSignal1 = QtCore.pyqtSignal(QtGui.QImage)
    VideoSignal3 = QtCore.pyqtSignal(QtGui.QImage) #---내가 추가한 --------------
    TextSignal1 = QtCore.pyqtSignal(QLabel)

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)
        
    @QtCore.pyqtSlot()
    def startVideo(self):

        global image

        run_video = True
        while run_video:

            
            ret, image = self.camera.read()
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            qt_image1 = QtGui.QImage(color_swapped_image.data,
                                    self.width,
                                    self.height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)
           
            self.VideoSignal1.emit(qt_image1)

            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(25, loop.quit) #25 ms
            loop.exec_()
    
    @QtCore.pyqtSlot()        
    def btn_message_clicked(self):
        if self.no_maksper > 0.9 :
            #kakao_token()  #이거는 한번만 보내라
            kakao_token_mesage_to_friends()

    @QtCore.pyqtSlot()
    def btn_Picture_clicked(self):
        global mask_per
        self.flag = 1 - self.flag


        ret, frame = self.camera.read()
        cv2.imwrite('webcam_1.jpg',frame)# 파일 생성
        color_swapped_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        img = cv2.imread('webcam_1.jpg')
        h, w = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
        facenet.setInput(blob)
        dets = facenet.forward()

        result_img = img.copy()

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
            
            self.no_maksper = nomask
            mask_per = ('%.2f%%' % (mask * 100))
            print(mask_per)
            picture_maskper = QLabel(mask_per)
            self.TextSignal1.emit(picture_maskper)

            if mask > nomask:
                color = (0, 200, 55)
                label = 'Mask %d%%' % (mask * 100)
            else:
                color = (150, 50, 0)
                #이 부분에 사진 저장 코드 들어가기
                cv2.imwrite('webcam_nomask.jpg',face)
                label = 'No Mask %d%%' % (nomask * 100)

            cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA) #사각형 만들기
            cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA) #텍스트 입히기
        
        cv2.imwrite('webcam_result.jpg',result_img)# 파일 생성
        img = cv2.imread('webcam_result.jpg') #다시 읽고

        color_swapped_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qt_image3 = QtGui.QImage(color_swapped_image.data,
            self.width,
            self.height,
            color_swapped_image.strides[0],
            QtGui.QImage.Format_RGB888)
        self.VideoSignal3.emit(qt_image3)
        
        #if nomask > 0.9 :
            
            #kakao_token()  #이거는 한번만 보내라
            #kakao_token_mesage_to_friends()
            #kakao_token_mesage_to_someone()


class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
    
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()
    
    def initUI(self):
        self.setWindowTitle('Test')

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()

class TextViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(TextViewer, self).__init__(parent)
        self.mask_percent = QLabel()
    
    def initUI(self):
        self.setWindowTitle('Test')

    @QtCore.pyqtSlot(QtWidgets.QLabel)
    def setText(self, mask_percent):
        print(mask_percent)
        self.update() 

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    #thread1 = QtCore.QThread()
    #thread1.start()
    vid = ShowVideo()
    #vid.moveToThread(thread1)
    

    image_viewer1 = ImageViewer()
    image_viewer3 = ImageViewer()
    text_viewer1 = TextViewer()

    vid.VideoSignal1.connect(image_viewer1.setImage)
    vid.VideoSignal3.connect(image_viewer3.setImage)
    vid.TextSignal1.connect(text_viewer1.setText)

    

    push_button1 = QtWidgets.QPushButton('Start')
    btn_Picture = QtWidgets.QPushButton('Picture')
    btn_message = QtWidgets.QPushButton('Message')
    #btn_Quit = QtWidgets.QPushButton('Quit')
    push_button1.clicked.connect(vid.startVideo)
    btn_Picture.clicked.connect(vid.btn_Picture_clicked)
    #btn_Quit.clicked.connect(QCoreApplication.instance().quit)
    btn_message.clicked.connect(vid.btn_message_clicked)
    
    #vertical_layout = QtWidgets.QVBoxLayout()
    #horizontal_layout = QtWidgets.QHBoxLayout()
    

    horizontal_layout.addWidget(image_viewer1)
    horizontal_layout.addWidget(text_viewer1)
    #horizontal_layout.addWidget(picture_maskper)
    horizontal_layout.addWidget(image_viewer3)
    
    vertical_layout.addLayout(horizontal_layout)
    vertical_layout.addWidget(push_button1)
    vertical_layout.addWidget(btn_Picture)
    vertical_layout.addWidget(btn_message)
    #vertical_layout.addWidget(btn_Quit)

    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(vertical_layout)

    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    sys.exit(app.exec_())

