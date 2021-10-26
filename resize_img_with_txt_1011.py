# 201621123 김현성
# cmd에서 작업할 것
# 이 디렉토리의 하위파일 안의 모든 txt파일들과 .jpg파일을 찾아서 txt파일의 좌표부분을 읽어들여
# 이미지의 부분을 자르고 class대로 저장함
# 이제부터 해야지 1011
import json
import os
import os.path
import glob
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

classes = ["0" ,"1" ,"2" ,"3"] #txt에서 내가 탐색할 클래스 이름들
                               #부츠 온, 오프, 헬멧 온, 오프
dir = "C:/Users/user/Desktop/cap_new/Yolo_mark/x64/Release/data/img/"

img_path = []
txt_path = []
img_names = [os.path.basename(x) for x  in glob.glob(r'**/*.jpg',recursive=True)] #img 파일이름만
txt_names = [os.path.basename(y) for y  in glob.glob(r'**/*.txt',recursive=True)] #json 파일이름만

def convert(size, box):
    #  (리사이즈 된 가로 중간점 * 원 이미지 가로 사이즈)
    #   - (리사이즈 된 가로범위 * 원래 이미지가로사이즈) / 2
    #   범위를 반으로 자르고 중간점에서 뺴면 시작점 좌표가 나옴
    x = (box[1] * size[0]) - ((box[3] * size[0]) / 2)
    #y 시작점 좌표도 동일
    y = (box[2] * size[1]) - ((box[4] * size[1]) / 2)
    w = box[3] * size[0]
    h = box[4] * size[1]
    #print(x, y, w, h) #여기서 어떤 작용을 해야 할 것 같은데 그게 뭘까?
    return (x, y, w, h)

def find_img_path():
    for filename in glob.iglob(r'**/*.jpg', recursive=True):
        if os.path.isfile(filename) :
            filename = dir + filename
            #print(filename)
            img_path.append(filename)

def find_txt_path():
    
    for txt_file in glob.iglob(r'**/*.txt', recursive=True):
        location = "C:/Users/user/Desktop/cap_new/Yolo_mark/x64/Release/data/img/"
        txt_path.append(location + txt_file)

    print(len(txt_path))
    

def resize_img():

    #img_path.append(filename)      #이미 find_img_path에서 집어넣음
            
 
    cutted_path = 'C:/Users/user/Desktop/cap_new/Yolo_mark/x64/Release/data/Cutted_Training_Image/'

    total_image = len(img_path)
    print("------------------------------------")
    print(total_image)
    index = 0
    num = 0
    for name in img_path :                          #이미지 경로에 있는걸 하나씩 돌면서
        print("----------이미지--------------")
        
         
        img = Image.open(img_path[index])  
        width, height = img.size                     # 가로 세로           
        with open(txt_path[index], 'r', encoding='UTF8') as f:  #여기서 bb값을 무한으로 주면서 bb값이 끝날 때 까지 이미지를 잘라서 저장
            
            for item1 in f.readline():
                # 여기서 한줄을 읽었으니까 그걸 ' '공백으로 잘라서 총 5개로 저장
                # 그걸 가지고 0번을 인덱스로 나머지를 좌표로 계산해야 하는데
                # 좌표 계산을 어떻게 해야할지 모르겟다잉
                
                box = []
                box = float(item1.split())
                print(box)
                #cls = item1["class"]

                #cls_id = classes.index(cls)
                box = item1["box"]
                print(box)
                bb = convert((width, height), box)
                
                cls = str(box[0])

                cutted_img = img.crop(bb) #bb에 들어있는 값으로 index에 따라서 사진을 자른다.
                cutted_img.save('%s%s'%(cutted_path + cls + '/', str(num) + img_names[index]))
                num = num + 1
                #print(num)
            #print(name + '   ' + str(index) + '/' + str(total_image))
        index = index + 1
        print(index)

if __name__ == '__main__':
    #resize_img(find_json())
    find_txt_path()
    find_img_path()
    resize_img()
