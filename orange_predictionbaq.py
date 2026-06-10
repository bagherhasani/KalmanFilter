from kalmanfilter import KalmanFilter

import cv2
from orange_detector import OrangeDetector


#feed the orange video to cv2
cap=cv2.VideoCapture("orange.mp4")


#Load Detector
od=OrangeDetector()

#Load Kalman Filter
kf=KalmanFilter()

while True:
    ret,frame = cap.read()
    if ret is False:
        break

    #showing the actual trajectory using red
    orange_bbox=od.detect(frame)
    x,y,x2,y2=orange_bbox

    centerX=int((x+x2)/2)
    centerY=int((y+y2)/2)

    #Predict and show with Kalman Filter
    predicted=kf.predict(centerX,centerY)

    cv2.circle(frame,(centerX,centerY),20,(0,0,255),4)

    cv2.circle(frame,(predicted[0],predicted[1]),20,(255,0,0),4)
    
    

  
    cv2.imshow("Frame",frame)

    #show frame
    key = cv2.waitKey(0) & 0xFF

    # Press q or ESC to exit
    if key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()