import cv2
import time

COLORS = [(0,255,255),(255,255,0), (0, 255, 0), (255,0,0)]

class_names = []
with open('yolo_files/coco2.names', "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#cap = cv2.VideoCapture("yolo_files/people2.mp4")#"yolo_files/street.mp4"
cap = cv2.VideoCapture("yolo_files/walking3.mp4")

net = cv2.dnn.readNet("yolo_files/yolov4.weights", "yolo_files/yolov4.cfg")

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

while True:
    _, frame = cap.read()
    start = time.time()
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)
    end = time.time()

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]

        label = f"{class_names[classid]}: {score}"

        x, y, w, h = box
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    fps_label = f"FPS: {round((1.0 / (end - start)), 2)}"

    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    cv2.imshow("detecao de video", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()