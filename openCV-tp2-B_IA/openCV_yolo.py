import cv2 as cv
import numpy as np
from wandb import Classes

# Carregar o algoritmo yolo
net = cv
net = cv.dnn.readNet("files/yolov3.weights", #fichiero que contém o modelo treinado para detetar os objetos nas imagens
                     "files/yolov3.cfg") #ficheiro de configuração
clasees = []
with open("files/coco.names", 'r') as f: #ficheiro que contém os nomes dos objetos que o algoritmo deteta
    classes = [line.strip() for line in f.readlines()] #guardados na variavel classes

layer_name = net.getLayerNames()
output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Carregar as Imagens
img = cv.imread("images/7.jpeg")
img = cv.resize(img, None, fx=0.4, fy=0.4)
height, width, channel = img.shape #altura e comprimento da imagem

# Detetar Objetos
blob = cv.dnn.blobFromImage(
    img, 0.00392, (416, 416), (0, 0, 0), True, crop=False) #Conversao da imagem em blob
                                                           #que extrai recursos da iamgem
                                                           #e redimensiona
net.setInput(blob)
outs = net.forward(output_layer)

# Mostra informação
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Deteção de objetos
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Coordenadas do retangulo
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)

font = cv.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]]) #variável reponsável por imprimir o tipo de objeto apresentado(coco.names)
        color = colors[i]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv.putText(img, label, (x, y + 30), font, 3, color, 3)

cv.imshow("IMG", img)
cv.waitKey(0)
cv.destroyAllWindows()