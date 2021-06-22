import cv2                                  #for capturing video from the webcam
import mediapipe as mp                      #for getting hand tracking coordinates
import time




class Hand_Detector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self, trackCon)
        self.mpDraw = mp.solutions.drawing_utils        #used to draw the hand landmarks


    def Find_Hands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)               #coverting img to rgb because hands module always takes img in rgb form
        result = self.hands.process(imgRGB)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)       #HAND_CONNECTIONS connects the coordinates

            # for id, lm in enumerate(handLms.landmark):
            #         h, w, c = img.shape
            #         cx, cy = int(lm.x*w), int(lm.y*h) 
            #         print(id, cx, cy)
            #         cv2.circle(img, (cx, cy), 8, (0, 0, 0), cv2.FILLED)

def main():
    cap = cv2.VideoCapture(0)
    c_time = 0
    p_time = 0

    while True:
        success, img = cap.read()
        c_time = time.time()                    #fetches the current time
        fps = 1/(c_time-p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (20, 80), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,0), 2)
        cv2.imshow("Image", img)                #used to display image in a window
        cv2.waitKey(1)                          #value 1 gives live video feed and value 0 gives images


if __name__=="__main__":
    main()