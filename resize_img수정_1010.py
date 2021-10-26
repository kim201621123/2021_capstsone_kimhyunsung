# cmd에서 작업할 것
# 본 파일과 하위파일 안의 모든 json파일들과 .jpg파일을 찾아서
# json파일의 box부분을 읽어들여
# 이미지의 부분을 자르고 class대로 저장함
# 완벽하게 작동함 1011 아니다 보니까 조금 헬멧이 안쓴거로 섞여 들어가던데?
# 많이는 아니고 조금 들어가긴 한다 왜지?
import json
import os
import os.path
import glob
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

classes = ["05" ,"06" ,"07" ,"08"] #json에서 내가 탐색할 클래스 이름들

dir = "C:/Users/user/Desktop/cap_new/03.공사현장안정장비인식_sample/"

img_path = []
json_path = []
img_names = [os.path.basename(x) for x  in glob.glob(r'**/*.jpg',recursive=True)] #img 파일이름만
json_names = [os.path.basename(x) for x  in glob.glob(r'**/*.json',recursive=True)] #json 파일이름만

def convert(size, box):
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    #print(x, y, w, h)
    return (x, y, w, h)

def find_img_path():
    for filename in glob.iglob(r'**/*.jpg', recursive=True):
        if os.path.isfile(filename) :
            filename = dir + filename
            #print(filename)
            img_path.append(filename)

def find_json_path():
    
    for json_file in glob.iglob(r'**/*.json', recursive=True):
        location = "C:\\Users\\user\\Desktop\\cap_new\\03.공사현장안정장비인식_sample\\"
        
        json_path.append(location + json_file)

    print(len(json_path))
    

def resize_img():

    #img_path.append(filename)      #이미 find_img_path에서 집어넣음
            
 
    cutted_path = 'C:/Users/user/Desktop/cap_new/03.공사현장안정장비인식_sample/Cutted_Training_Image/'

    total_image = len(img_path)
    print("------------------------------------")
    print(total_image)
    index = 0
    num = 0
    for name in img_path :                          #이미지 경로에 있는걸 하나씩 돌면서
        print("----------이미지--------------")
        print(img_path[index])
        #print("------------------------------------")
         
        img = Image.open(img_path[index])  
        #for num in                                 #여기서 for문으로 돌면서 bb의 한 줄이 끝날 때 까지
        with open(json_path[index], 'r', encoding='UTF8') as f:  #여기서 bb값을 무한으로 주면서 bb값이 끝날 때 까지 이미지를 잘라서 저장
            datas = json.load(f)

            data0 = datas["image"]

            data = datas["annotations"]

            json_filename = data0["filename"]
        
            width, height = data0["resolution"]
            #print(width, height)
            #width = 1920
            #height = 1080
        
        print("----------box--------------")
        for item1 in data:
            print(json_names[index][:-5])  #   .json을 뺴고
            #outfile = open('%s.txt' % (json_filenames[index][:-4]), 'a+')
        
            cls = item1["class"]
        
            if cls not in classes:
                #bb[i] = (0,0,0,0)  #만약 bb값이 0000뿐이면 그냥 지나가기
                bb = 0,0,0,0        #json안에 들어있는 클래스랑 내가 찾는 거랑 다르면 0 ,0,0,0 넣고 사진 안짜르고 저장도 안하고 다음 클래스찾기로 넘어감
                cls = "00"
                continue

            #cls_id = classes.index(cls)
            box = item1["box"]
            print(box)
            bb = convert((width, height), box)
           
            
            cutted_img = img.crop(bb) #bb에 들어있는 값으로 index에 따라서 사진을 자른다.
            cutted_img.save('%s%s'%(cutted_path + cls + '/', str(num) + img_names[index]))
            num = num + 1
            #print(num)
        #print(name + '   ' + str(index) + '/' + str(total_image))
        index = index + 1
        print(index)

if __name__ == '__main__':
    #resize_img(find_json())
    find_json_path()
    find_img_path()
    resize_img()
