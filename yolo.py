import cv2
import numpy as np
import matplotlib.pyplot as plt

net=cv2.dnn.readNet("yolov3 (1).weights","yolov3.cfg")
classes=[]
with open("coco.names","r") as f:
    for line in f.readlines():
        classes.append(line.strip())
print(classes)
len(classes)
img=cv2.imread("IMG_4934.JPG")
img=cv2.resize(img,(640,640))
height,width,channels=img.shape
print(height,width,channels)
blob=cv2.dnn.blobFromImage(img,1/255.0,(640,640),(0,0,0),True,crop=False)
net.setInput(blob)
layer_names=net.getLayerNames()
print(layer_names)
output_layers=[]
for i in net.getUnconnectedOutLayers():
    output_layers.append(layer_names[i-1])
print(output_layers)
blobdata=net.forward(output_layers)
print(blobdata)
colors=np.random.uniform(0,255,size=(80,3))
class_ids=[]
confidences=[]
boxes=[]
for x in blobdata:
    for detection in x:
        scores=detection[5:]
        class_id=np.argmax(scores)
        confidence=scores[class_id]
        if confidence>0.5:
            center_x=int(detection[0]*width)
            center_y=int(detection[1]*height)
            w=int(detection[2]*width)
            h=int(detection[3]*height)
            x=int(center_x-(w/2))
            y=int(center_y -(h/2))
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
indexes=cv2.dnn.NMSBoxes(boxes, confidences,0.5,0.4)
for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h=boxes[i]
        label=classes[class_ids[i]]
        confidence_score=int(confidences[i]*100)
        color=colors[i]
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        str1='%s %d'%(label,confidence_score)
        cv2.putText(img,str1,(x-10,y+75),cv2.FONT_HERSHEY_DUPLEX,1,color,2)
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()