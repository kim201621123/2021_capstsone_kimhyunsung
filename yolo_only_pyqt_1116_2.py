# 2021/11/01 01:45 201621123 김현성 작성중 cmd에서 실행해라
# 나중에는 불필요한 주석들 모두 지우기
# 2021 11 02 03:43 201621123 김현성 중간작성 완료 - 리스트에 저장까지
# 2021 11 07 18:29 201621123 김현성 엑셀에 파일로 저장까지 완료
# 모바일넷하고 카페모델 빼고 그냥 욜로로만
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
import os
import sys
import tensorflow as tf
import time
import openpyxl
import winsound as sd
from playsound import playsound
import pygame
from gtts import gTTS
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QLabel, QHBoxLayout, QVBoxLayout ,QAction, QTableWidget,QTableWidgetItem
from PyQt5.QtGui import *
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import pyqtSlot
from PyQt5 import sip


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices()) #텐서플로우가 gpu를 잡는지 검사

facenet = cv2.dnn.readNet('models/res10_300x300_ssd_iter_140000.caffemodel','models/deploy.prototxt' )
#model = load_model('models/mask_detector.model')

#model = tf.keras.models.load_model("mask-no-mask.h5")
#model = load_model('face_recognization_e40_1106.h5') #제일 오류율이 낮은 40으로 진행함

#model = load_model('face_recognization_e20_1111_3.h5')

YOLO_net = cv2.dnn.readNet("yolov4-custom_1114_10_last.weights","yolov4-custom_1114_10.cfg")
#YOLO_net = cv2.dnn.readNet("yolov4-custom_1116_10_final.weights","yolov4-custom_1116_10.cfg")

#YOLO_net = cv2.dnn.readNet("yolov4-custom_last.weights","yolov4-custom.cfg")
#1029 해결 opencv를 빌드했기 때문에 cmd에서 사용하세요
YOLO_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
YOLO_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = [[],[]]
with open("./data/obj_1114_10.names", "r") as f:
#with open("./data/obj_1116_10.names", "r") as f:    
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
#output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()] 잘못됨
output_layers = [layer_names[i-1] for i in YOLO_net.getUnconnectedOutLayers()]


# ---------------------------엑셀--------------------------------
wb = openpyxl.Workbook()    #엑셀파일 생성
sheet1 = wb['Sheet']
sheet1.title = '출근인원'
time_for_name = datetime.now()
excel_file_name = '인원 리스트 '+str(str(time_for_name.year) +"년 " + str(time_for_name.month) +"월 "+str(time_for_name.day) +"일 "+str(time_for_name.hour)+"시 "+str(time_for_name.minute)+"분")+'.xlsx'
wb.save(excel_file_name)       #엑셀파일 저장
# --------------------------------------------------------------
vertical_layout = QtWidgets.QVBoxLayout()
horizontal_layout = QtWidgets.QHBoxLayout()


class ShowVideo(QtCore.QObject):
    
    #f = open("./log" + str(time.time()) + ".txt",'w') # 공사장 출입 로그를 txt로 저장
    #ValueError: I/O operation on closed file. 고칠 것
    prevTime = 0

    formal_yolo_class = ""
    formal_time = 0

    formal_time_yolo = 0
    formal_time_mobilenet = 0
    flag = 0

    #label_mo = ""
    label_yo = ""
    tmp_human = []
    human_table = []
    human_table_pre_count = 0

    text_voice_g = ""       #mp3 파일 이름


    yolo_condidence = 0

    yolo_flag = 0

    camera = cv2.VideoCapture(0)

    ret, image = camera.read()
    height, width = image.shape[:2]

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
            #ret, image = self.camera.read()
            
            now_time = datetime.now()       #현재 시간

            curTime = time.time()

            sec = curTime - self.prevTime
            self.prevTime = curTime

            self.flag = 1 - self.flag

            self.yolo_flag = 0
            
            ret, frame = self.camera.read()
            #cv2.imwrite('webcam_1.jpg',frame)# 파일 생성
            

            #frame = cv2.imread('webcam_1.jpg')
            h, w = frame.shape[:2]
            
            try:
                blob = cv2.dnn.blobFromImage(frame, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
            except:
                continue
            
            facenet.setInput(blob)
            dets = facenet.forward() #추론
            

            faces = []

            for i in range(dets.shape[2]):
                confidence = dets[0, 0, i, 2]
                if confidence < 0.9:
                    continue
                
                #x1 = int(dets[0, 0, i, 3] * w)
                #y1 = int(dets[0, 0, i, 4] * h)
                #x2 = int(dets[0, 0, i, 5] * w)
                #y2 = int(dets[0, 0, i, 6] * h)

                #x1 = int(dets[0, 0, i, 3] * w * (0.95)) #이마도 빼고 얼굴만 자르기 1101 확인해봤는데 잘 나옴
                #y1 = int(dets[0, 0, i, 4] * h * (1.2))
                #x2 = int(dets[0, 0, i, 5] * w * (1.03))
                #y2 = int(dets[0, 0, i, 6] * h * (1))


                x1 = int(dets[0, 0, i, 3] * w * (0.7)) # 다시 확장된 얼굴
                y1 = int(dets[0, 0, i, 4] * h * (0.5))
                x2 = int(dets[0, 0, i, 5] * w * (1.2))
                y2 = int(dets[0, 0, i, 6] * h / (1))
                
                face = frame[y1:y2, x1:x2]
                faces.append(face)
                self.yolo_flag = 1
                #cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=(0,0,255), lineType=cv2.LINE_AA) #사각형 만들기

            #print("욜로 중간 현재 값"+str(self.yolo_flag))
            if self.yolo_flag == 1:     #화면에 얼굴이 있으면    문제 해결!
                #화면에 사람 전부 체크할 꺼면 여기서 부터 밑 코드 한 칸 들여쓰기 
                try:
                    blob = cv2.dnn.blobFromImage(face, 0.00392, (608, 608), (0, 0, 0),
                    True, crop=False)
                except:
                    continue
                YOLO_net.setInput(blob)
                outs = YOLO_net.forward(output_layers)

                class_ids = []
                confidences = []
                boxes = []
                
                for out in outs:

                    for detection in out:

                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if confidence > 0.98:       #98%
                            # Object detected
                            center_x = int(detection[0] * w)
                            center_y = int(detection[1] * h)
                            dw = int(detection[2] * w)
                            dh = int(detection[3] * h)
                            # Rectangle coordinate
                            x = int(center_x - dw / 2)
                            y = int(center_y - dh / 2)
                            boxes.append([x, y, dw, dh])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)
            
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        self.label_yo = str(classes[class_ids[i]])
                        score = confidences[i]

                        cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=(0,250,150), lineType=cv2.LINE_AA) #초록사각형 만들기
                        #print(self.label_yo[:-2]+"  인증되었습니다.")
                        b,g,r,a = 0,255,150,10
                        fontpath = "fonts/gulim.ttc"
                        font = ImageFont.truetype(fontpath, 30)
                        img_pil = Image.fromarray(frame)
                        draw = ImageDraw.Draw(img_pil)
                        draw.text((x1 + (x2-x1)/2. -25, y2 +20),  self.label_yo[:-2], font=font, fill=(b,g,r,a),
                            stroke_width=2,stroke_fill="black")
                        frame = np.array(img_pil)
                        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                        #cv2.putText(frame, self.label_yo, (x, y - 20), cv2.FONT_ITALIC, 0.8, (0, 0, 255),thickness=2, lineType=cv2.LINE_AA)
                    self.yolo_flag = 2        # 욜로 디텍팅 과정을 거쳤다면 2

            

            #print("현재 yolo_flag 값  :" + str(self.yolo_flag))
            
            if self.yolo_flag == 0:     #얼굴도 없는 경우
                print("-------빈화면------")
                self.yolo_flag = 0
            if self.yolo_flag == 1:     #얼굴은 있는데 욜로 디텍팅이 안되었다면? 모르는 사람이면?
                print("인증되지 않음")
                cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=(0,0,225), lineType=cv2.LINE_AA) #빨간사각형 만들기
                
                b,g,r,a = 0,0,255,10
                fontpath = "fonts/gulim.ttc"
                font = ImageFont.truetype(fontpath, 50)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text((x1 - 30, y2 +20),  "식별되지 않음", font=font, fill=(b,g,r,a),
                    stroke_width=2,stroke_fill="black")
                frame = np.array(img_pil)
                p = 6
                if self.formal_time_mobilenet == 0: #맨처음 식별되지 않았다면
                    self.formal_time_mobilenet = curTime#현재 시간을 넣고
                if curTime - self.formal_time_mobilenet > p :   
                    #self.beepsound(0)
                    self.tts_google("식별되지 않은 인원입니다." ," ")       #gtts로 mp3를 만들고
                    self.mp3_play(self.text_voice_g)                        # pygame으로 재생
                    self.formal_time_mobilenet = 0
                self.yolo_flag = 0
                    #모르는 사람이면 제일 높은 소리
                #cv2.putText(frame, text="식별되지 않음", org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,0,225), thickness=2, lineType=cv2.LINE_AA) #텍스트 입히기
            if self.yolo_flag == 2:     #얼굴도 있고 욜로도 값이 나왔다면
                frame = self.is_that_you(self.label_yo, curTime, score, frame)
                self.yolo_flag = 0

            self.yolo_flag = 0
            color_swapped_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_image1 = QtGui.QImage(color_swapped_image.data,
            self.width,
            self.height,
            color_swapped_image.strides[0],
            QtGui.QImage.Format_RGB888)
            self.VideoSignal1.emit(qt_image1)
            fps = 1/(sec)
                #print( "Time {0} " . format(sec))
                #print( "Estimated fps {0} " . format(fps))

                
                #self.is_that_you(self.label_yo, curTime)
                #self.createTable()
                #with open("human_table.pickle","rb") as fr:
                        #data_human_table=pickle.load(fr)
                #print(data_human_table)
                
            
            #horizontal_layout.addWidget(self.tableWidget)
            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(25, loop.quit) #25 ms
            loop.exec_()
                
            


    
    #f.close()

    def tts_google(self, yolo_text ,yolo_name):
        text_voice = yolo_text + yolo_name
        self.text_voice_g = text_voice +".mp3"      #text_voice_g에 파일이름 저장
        tts = gTTS(text=text_voice, lang='ko')
        tts.save(text_voice +".mp3")
        

    def mp3_play(self, mp3_name):
        music_file = mp3_name
        #freq = 24000    # sampling rate, 44100(CD), 16000(Naver TTS), 24000(google TTS)
        freq = 24000
        bitsize = -16   # signed 16 bit. support 8,-8,16,-16
        channels = 1    # 1 is mono, 2 is stereo
        buffer = 4096   # number of samples (experiment to get right sound)

        # default : pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096)
        pygame.mixer.init(freq, bitsize, channels, buffer)
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.play()

        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick(30)
        pygame.mixer.quit()


    def beepsound(self, flag):
        if flag == 0:
            fr = 3000   #좀 더 날카로운 소리가 나겠지
        if flag == 1:
            fr = 2000    # range : 37 ~ 32767
        if flag == 2:
            fr = 1000
        if flag == 3:
            fr = 600
        du = 1000     # 1000 ms == 1second
        sd.Beep(fr, du) # winsound.Beep(frequency, duration)
                

    def is_that_you(self, yolo_class, cur_time, yolo_score,frame): #작동함
        
        self.yolo_flag = 0

        self.tmp_human = []

        if self.formal_yolo_class != yolo_class: #이전 사람과 같지 않다면
            self.formal_yolo_class = yolo_class #사람 업데이트
            self.formal_time = cur_time         # 시간 업테이트
            #print("사람 업데이트 되었습니다")
            #self.beepsound(2)

            return frame
        

        n = 1   #같은 사람 체크 시간
        if self.formal_yolo_class == yolo_class: #사람이 이전 사람과 같다면?
            #print("같은사람입니다")
            
            if cur_time - 6 > self.formal_time:     #일정시간 이상 이전 사람과 같다 판정되면
                #이제 끝에 자리가 n인지 h인지 판단해서 넣기
                #print("일정시간")
                
                if len(self.human_table) > 0 :     #아예 처음이 아니고
                    for i in range(len(self.human_table)):      #TypeError: 'int' object is not iterable 
                        # 리스트에 사람 and 헬멧을 쓰고있는게 맞다면  == 헬멧을 안썻을 때는 그냥 들어감
                        if (self.human_table[-(i+1)][-3] == yolo_class[:-2]) and self.human_table[-(i+1)][-2] == "True" :   #사람이 리스트에 있고 / 뒤에서부터 사람만 센다
                            #print("리스트에 이 사람이 있어요")
                            d = datetime.now()
                            time_tmp=int(d.minute)
                            d2 = self.human_table[-(i+1)][-4]       # 시간이 제일 앞에 오니까
                            time_tmp2 = int(d2.minute)

                            

                            if abs(time_tmp - time_tmp2) < n :    #리스트에 시간이 n분이상 차이 안나면
                                #print("일정 시간 이후에 다시 오세요")
                                b,g,r,a = 0,0,255,255
                                fontpath = "fonts/gulim.ttc"
                                font = ImageFont.truetype(fontpath, 30)
                                img_pil = Image.fromarray(frame)
                                h, w = frame.shape[:2]
                                draw = ImageDraw.Draw(img_pil)
                                draw.text((0, 0), yolo_class[:-2]+ "  " +str(n)+"분 이내 측정된 인원입니다.",
                                    font=font, fill=(b,g,r,a), stroke_width=2,stroke_fill="black")
                                frame = np.array(img_pil)
                                
                                m = 6
                                if self.formal_time_yolo == 0: #맨처음 식별되지 않았다면
                                    self.formal_time_yolo = cur_time#현재 시간을 넣고
                                    #self.beepsound(1)
                                    self.tts_google( str(n)+"분 이내 측정된 인원입니다." , yolo_class[:-2])       #gtts로 mp3를 만들고
                                    self.mp3_play(self.text_voice_g)                        # pygame으로 재생
                                if cur_time - self.formal_time_yolo > m :   #일정 시간 이후 시간 울리게
                                    self.formal_time_yolo = 0
                                return frame

                #if self.human_table.count(human) == 0: 
                # 헬멧이던 노헬멧이던 리스트에 넣는건 다 넣는다

                #self.tmp_human = []
                daytime_now = datetime.now() #시간
                #self.tmp_human.append(cur_time)# 
                self.tmp_human.append(daytime_now)  #시간   넣기
                
                
                self.tmp_human.append(yolo_class[:-2])   #이름만   넣기
                if yolo_class[-1:] == "n":
                    self.tmp_human.append("False")      # 끝자리가 n이면 False
                else:
                    self.tmp_human.append("True")       # h True
                self.tmp_human.append(yolo_score)   #정확도 넣기
                
                        
                wb = openpyxl.load_workbook(excel_file_name)   #엑셀파일 로드
                sheet1 = wb.active      #시트 1
                
                
                self.human_table.append(self.tmp_human) #리스트에 넣기


                sheet1.append(self.tmp_human)    #엑셀 시트 1에 넣기    시간 사람 정확도
                wb.save(excel_file_name)       #엑셀 수정내용 저장
                
                #print("리스트에 넣었습니다.") #
                print("----------------------------------------")
                self.tts_google("리스트에 넣었습니다." , yolo_class[:-2])       #gtts로 mp3를 만들고
                self.mp3_play(self.text_voice_g)                        # pygame으로 재생

                text_draw = ""

                fontpath = "fonts/gulim.ttc"
                b,g,r,a = 225,0,255,10
                text_draw = yolo_class[:-2]+ "  " +" 리스트에 넣었습니다."
                
                    
                #self.tts_google(human)
                if yolo_class[-1:] == "n":   # 인물이 헬멧을 안쓰고있으면
                    #print("헬멧 쓰십셔")
                    b,g,r,a = 255,200,50,10
                    fontpath = "fonts/gulim.ttc"
                    text_draw = text_draw + "\n 헬멧을 착용하세요."
                    #self.beepsound(2)
                    self.tts_google( "헬멧을 착용하세요." , yolo_class[:-2])       #gtts로 mp3를 만들고
                    self.mp3_play(self.text_voice_g)                        # pygame으로 재생
                    
                font = ImageFont.truetype(fontpath, 30)
                img_pil = Image.fromarray(frame)
                h, w = frame.shape[:2]
                draw = ImageDraw.Draw(img_pil)
                draw.text((0, 20), text_draw, font=font, fill=(b,g,r,a), stroke_width=2,stroke_fill="black")

                #draw.text((x, y), text, fill=color, font=font, stroke_width=2,
                #stroke_fill="black")

                frame = np.array(img_pil)
                self.beepsound(3)
                print("----------------------------------------")
                self.formal_time = cur_time         # 시간 업테이트
            #print(self.human_table)
            #print(self.tmp_human)
                    
        return frame

   
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
    #image_viewer3 = ImageViewer()
    #text_viewer1 = TextViewer()

    vid.VideoSignal1.connect(image_viewer1.setImage)
    #vid.VideoSignal3.connect(image_viewer3.setImage)
    #vid.TextSignal1.connect(text_viewer1.setText)

    

    push_button1 = QtWidgets.QPushButton('Start')
    #btn_Picture = QtWidgets.QPushButton('Picture')
    #btn_Quit = QtWidgets.QPushButton('Quit')
    push_button1.clicked.connect(vid.startVideo)
    #btn_Picture.clicked.connect(vid.btn_Picture_clicked)
    #btn_Quit.clicked.connect(QCoreApplication.instance().quit)


    #vertical_layout = QtWidgets.QVBoxLayout()
    #horizontal_layout = QtWidgets.QHBoxLayout()
    

    horizontal_layout.addWidget(image_viewer1)
    #horizontal_layout.addWidget(text_viewer1)
    #horizontal_layout.addWidget(picture_maskper)
    #horizontal_layout.addWidget(image_viewer3)
    #horizontal_layout.addWidget(self.tableWidget)


    vertical_layout.addLayout(horizontal_layout)
    vertical_layout.addWidget(push_button1)
    #vertical_layout.addWidget(btn_Picture)
    #vertical_layout.addWidget(btn_Quit)

    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(vertical_layout)

    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    sys.exit(app.exec_())
