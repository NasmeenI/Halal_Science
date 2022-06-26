import cv2
import numpy as np
from tkinter import *

def apple():
    root = Tk()
    root.title("Apple")
    mylabel1 = Label(root, text="Apple", fg="red", font=20, bg="yellow").place(x=210, y=0)
    mylabel2 = Label(root, text="แคลลอรี่       =    52  kcal", font=10).place(x=140, y=30)
    mylabel3 = Label(root, text="ไขมันทั้งหมด  =    0.2 g", font=10).place(x=140, y=60)
    mylabel4 = Label(root, text="โปรตีน          =    0.3 g", font=10).place(x=140, y=90)
    root.geometry("500x200+500+200")
    root.mainloop()

def banana():
    root = Tk()
    root.title("Banana")
    mylabel1 = Label(root, text="Banana", fg="red", font=20, bg="yellow").place(x=210, y=0)
    mylabel2 = Label(root, text="แคลลอรี่       =    88  kcal", font=10).place(x=140, y=30)
    mylabel3 = Label(root, text="ไขมันทั้งหมด  =    0.3 g", font=10).place(x=140, y=60)
    mylabel4 = Label(root, text="โปรตีน          =    1.1 g", font=10).place(x=140, y=90)
    root.geometry("500x200+500+200")
    root.mainloop()

def pizza():
    root = Tk()
    root.title("Pizza")
    mylabel1 = Label(root, text="Pizza", fg="red", font=20, bg="yellow").place(x=210, y=0)
    mylabel2 = Label(root, text="แคลลอรี่       =    266  kcal", font=10).place(x=140, y=30)
    mylabel3 = Label(root, text="ไขมันทั้งหมด  =    10  g", font=10).place(x=140, y=60)
    mylabel4 = Label(root, text="โปรตีน          =    11  g", font=10).p
    root.geometry("500x200+500+200")
    root.mainloop()

net = cv2.dnn.readNet('config_object/yolov3.weights' ,'config_object/yolov3.cfg')
classes = []
with open('config_object/coco.names' ,'r') as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture('Video/apple.mp4')
#cap = cv2.VideoCapture('rtsp://192.168.100.154:8080/h264_pcm.sdp')

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layersOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    ck = False

    for output in layersOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                if class_id != 47:
                    continue
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                ck = True

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(2, 255, size=(len(boxes), 3))
    font = cv2.FONT_HERSHEY_PLAIN
    label = str(" ")

    if ck == True:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

    cv2.imshow('VDO', img)
    if(label == "apple"):
        apple()
    if(label == "banana"):
        banana()
    if(label == "pizza"):
        pizza()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()