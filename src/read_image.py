import numpy as np
import cv2

cap = cv2.VideoCapture(0)

cap.set(3,1280)
cap.set(4,960)

cnt = 0
while True:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cnt += 1
    if cnt % 100 == 0:
        print(cnt)

cap.release()
cv2.destroyAllWindows()
