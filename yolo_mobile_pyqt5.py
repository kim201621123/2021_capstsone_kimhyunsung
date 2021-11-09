# 2021/11/01 01:45 201621123 김현성 작성중 cmd에서 실행해라
# 나중에는 불필요한 주석들 모두 지우기
# 2021 11 02 03:43 201621123 김현성 중간작성 완료 - 리스트에 저장까지
# 2021 11 07 18:29 201621123 김현성 엑셀에 파일로 저장까지 완료
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import sys
import tensorflow as tf
import time
import pickle
import openpyxl
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
model = load_model('face_recognization_e40_1106.h5') #제일 오류율이 낮은 40으로 진행함

YOLO_net = cv2.dnn.readNet("yolov4-custom_1030_last.weights","yolov4-custom_1030.cfg")
#YOLO_net = cv2.dnn.readNetFromDarknet("./yolov4-custom_last.weights","./yolov4-custom.cfg")
#1029 해결 opencv를 빌드했기 때문에 cmd에서 사용하세요
YOLO_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
YOLO_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = [[],[]]
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
#output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()] 잘못됨
output_layers = [layer_names[i-1] for i in YOLO_net.getUnconnectedOutLayers()]

# ---------------------------엑셀--------------------------------
wb = openpyxl.Workbook()    #엑셀파일 생성
sheet1 = wb['Sheet']
sheet1.title = '출근인원'
wb.save('test3.xlsx')       #엑셀파일 저장
# --------------------------------------------------------------
vertical_layout = QtWidgets.QVBoxLayout()
horizontal_layout = QtWidgets.QHBoxLayout()


class ShowVideo(QtCore.QObject):
    
    #f = open("./log" + str(time.time()) + ".txt",'w') # 공사장 출입 로그를 txt로 저장
    #ValueError: I/O operation on closed file. 고칠 것
    prevTime = 0

    formal_human = ""
    formal_helmet = ""
    formal_time = 0

    flag = 0

    label_mo = ""
    label_yo = ""

    human_table = []
    human_table_pre_count = 0



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
            ret, image = self.camera.read()
            
            
            curTime = time.time()

            sec = curTime - self.prevTime
            self.prevTime = curTime

            self.flag = 1 - self.flag


            ret, frame = self.camera.read()
            cv2.imwrite('webcam_1.jpg',frame)# 파일 생성
            

            img = cv2.imread('webcam_1.jpg')
            h, w = img.shape[:2]

            blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
            facenet.setInput(blob)
            dets = facenet.forward() #추론
            

            faces = []

            for i in range(dets.shape[2]):
                confidence = dets[0, 0, i, 2]
                if confidence < 0.5:
                    continue

                #x1 = int(dets[0, 0, i, 3] * w)
                #y1 = int(dets[0, 0, i, 4] * h)
                #x2 = int(dets[0, 0, i, 5] * w)
                #y2 = int(dets[0, 0, i, 6] * h)

                x1 = int(dets[0, 0, i, 3] * w * (0.95)) #이마도 빼고 얼굴만 자르기 1101 확인해봤는데 잘 나옴
                y1 = int(dets[0, 0, i, 4] * h * (1.2))
                x2 = int(dets[0, 0, i, 5] * w * (1.03))
                y2 = int(dets[0, 0, i, 6] * h * (1))

                #밑의 계산은 음수가 되거나 높이가 너비보다 커지면 오류가 생긴다고 해서 추가함
                #확실히 추가하니까 갑작스러운 오류는 생기지 않는듯 - 대신 얼굴이 너무 가까이 가면
                #계산시간이 길어진다. 크기가 크면 처리하기 힘들어지나보다.
                #02:36 갑자기 리사이즈 오류라고 안되는데 빼봐야겠다
                #빼도 오류가 생기는데 어디선가 잘못된게 틀림없다
                #밑에서 오류가 생겼던거다 그런데 이번에는 높이-너비 저기서 계산방해를 일으키는것 같다
                #저거 빼고 이리저리 움직여봐도 오류가 생기진 않았으니 빼고 해보자
                if x1 < 0:
                    x1 = x1 + 20
                if x2 < 0:
                    x2 = x2 + 20
                if y1 < 0:
                    y1 = y1 + 20
                if y2 < 0:
                    y2 = y2 + 20
                
                #if y2-y1 > x2-x1:
                    #continue
                
                face = img[y1:y2, x1:x2]
                faces.append(face)

            #---------------------mask detect ---------------------------------------------
            for i, face in enumerate(faces):
                face_input = cv2.resize(face, dsize=(224, 224))
                face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
                face_input = preprocess_input(face_input)
                face_input = np.expand_dims(face_input, axis=0)

                face1, face2, face3 = model.predict(face_input).squeeze()# 학습 할 때 dad, me mom 순으로 학습함

                #하드코딩으로 짠 거 나중에 고쳐라 - 더 좋은 방법이 있겠지 1102 02:13 max로 할 수 있을것같다.
                if (face1 > face2) and (face1 > face3):
                    color = (0, 200, 55)
                    self.label_mo = 'face1 %d%%' % (face1 * 100)
                if (face2 > face1) and (face2 > face1):
                    color = (150, 50, 0)
                    self.label_mo = 'face2 %d%%' % (face2 * 100)
                if (face3 > face2) and (face3 > face2):
                    color = (100, 100, 100)
                    self.label_mo = 'face3 %d%%' % (face3 * 100)
                
                if (0.9 > face1) and (0.9 > face2) and (0.9 > face3):#모르는 사람이면?
                    #침입자 경고 - 등록되지 않은 사람 - 한국어는 변환이 안되더라? 인코딩을 utf-8로 직접 줘야하나?
                    cv2.putText(img, text="unidentified", org=(x1, y2 + 45), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA) #텍스트 입히기
                    continue
                #self.picture_maksper = QLabel('%.2f%%' % (mask * 100))
                #self.vbox.addWidget(self.picture_maksper)
            
                #사각형은 넣지 않는다
                #cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA) #사각형 만들기
                #cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA) #텍스트 입히기
                cv2.putText(img, text=self.label_mo, org=(x1, y2 + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA) #텍스트 입히기

                x1 = int(dets[0, 0, i, 3] * w / (0.95) * (0.9)) #이마도 빼고 얼굴만 자르기 1101 확인해봤는데 잘 나옴
                y1 = int(dets[0, 0, i, 4] * h / (1.2) * (0.65))
                x2 = int(dets[0, 0, i, 5] * w / (1.03) * (1.1))
                y2 = int(dets[0, 0, i, 6] * h / (1))
                
                face = img[y1:y2, x1:x2]
                
                blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0),
                True, crop=False)
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

                        if confidence > 0.5:
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


                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)
                        cv2.putText(img, self.label_yo, (x, y - 20), cv2.FONT_ITALIC, 0.5, 
                        (255, 255, 255), 1)


                color_swapped_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qt_image1 = QtGui.QImage(color_swapped_image.data,
                self.width,
                self.height,
                color_swapped_image.strides[0],
                QtGui.QImage.Format_RGB888)
                self.VideoSignal1.emit(qt_image1)
            fps = 1/(sec)
            #print( "Time {0} " . format(sec))
            #print( "Estimated fps {0} " . format(fps))
            
           
            self.is_that_you(self.label_mo, self.label_yo, curTime)
            #self.createTable()
            #with open("human_table.pickle","rb") as fr:
                       #data_human_table=pickle.load(fr)
            #print(data_human_table)
            
            
            #horizontal_layout.addWidget(self.tableWidget)
            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(25, loop.quit) #25 ms
            loop.exec_()
    #f.close()

        

    def is_that_you(self, human, helmet, cur_time): #작동함

        

        if self.formal_human != human: #이전 사람과 같지 않다면
            self.formal_human = human #사람을 업데이트 하고
            #self.formal_time = cur_time #이전시간을 업데이트 한다
            #print("이전사람이 아닙니다")
        
        if self.formal_human == human: #사람이 이전 사람과 같다면?
            #print("같은사람입니다")
            if cur_time - 3 > self.formal_time:     #일정시간 이상 이전 사람과 같다 판정되면
                #print("3초간 같은 사람이었습니다.")
                self.formal_time = cur_time
                #if self.human_table.count(human) == 0: 
                if helmet == "helmet on":   # 인물이 헬멧을 쓰고 있다면
                

                    tmp_human = []
                    tmp_human.append(human)
                    tmp_human.append(helmet)
                    #tmp_human.append(cur_time)
                    print(self.human_table.count(tmp_human)) #처음은 0로 나오고 다음부터는 1로 나와야 함
                    if self.human_table.count(tmp_human) == 0: #해결! 같은사람이 2번 들어가는 일은 없음 /시간은 계속 달라져서 이렇게 하면 안될듯
                        #f.write(human+ " " +helmet + "\n")
                            
                        wb = openpyxl.load_workbook('test3.xlsx')   #엑셀파일 로드
                        sheet1 = wb.active      #시트 1
                        
                        self.human_table_pre_count = self.human_table_pre_count + 1
                        self.human_table.append(tmp_human) #리스트에 넣기       아직 시간이 들어가지 않음
                        
                        
                        
                        with open("human_table.pickle","wb") as fw:
                            #pickle.dump(self.human_table, fw)
                            pickle.dump(tmp_human, fw)      #시간 안들어간 채로 피클에 저장
                        
                        with open("human_table.pickle","rb") as fr:
                            data_human_table=pickle.load(fr)    #피클에서 읽어옴
                        
                        print(tmp_human)
                        tmp_human.append(cur_time)
                        sheet1.append(tmp_human)    #엑셀 시트 1에 넣기
                        wb.save('test3.xlsx')       #엑셀 수정내용 저장
                        print(data_human_table)
                        print("리스트에 넣었습니다.") #self.human리스트에 사람과 헬멧 착용여부가 들어감
                        print("----------------------------------------")
                        print("----------------------------------------")
                    #print(self.human_table)
                    #print(tmp_human)
                        
                

   
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

