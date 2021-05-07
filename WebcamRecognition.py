import cv2

classifier = cv2.CascadeClassifier ('clasificators/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)
print(cv2.__file__)
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = classifier.detectMultiScale(gray,scaleFactor = 1.5, minNeighbors = 5)
    for (x,y,w,h) in face:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        img_item = "my_image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (0,255,0)
        width =2
        coord_x = x+w
        coord_y = y+h
        cv2.rectangle(frame,(x,y),(coord_x,coord_y),color,width)

    cv2.imshow('Fereastra Video', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
