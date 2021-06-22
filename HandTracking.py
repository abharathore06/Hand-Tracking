import cv2                                  #for capturing video from the webcam
import mediapipe as mp                      #for getting hand tracking coordinates
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils        #used to draw the hand landmarks
c_time = 0
p_time = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)               #coverting img to rgb because hands module always takes img in rgb form
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) 
                print(id, cx, cy)
                #cv2.circle(img, (cx, cy), 8, (0, 0, 0), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)       #HAND_CONNECTIONS connects the coordinates

    c_time = time.time()                    #fetches the current time
    fps = 1/(c_time-p_time)
    p_time = c_time

    cv2.putText(img, str(int(fps)), (20, 80), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,0), 2)
    cv2.imshow("Image", img)                #used to display image in a window
    cv2.waitKey(1)                          #value 1 gives live video feed and value 0 gives images
