import numpy as np
import cv2
import mediapipe as mp
import time

# Setting the video feed size
wCam, hCam = 640, 480

# Initiaizing video stream
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Setting default values for prediction function execution
mode = False
maxHands = 2
detectionCon = 0.5
trackCon = 0.5

# Initialize the object for calling hand prediction from mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(mode, maxHands,detectionCon, trackCon)
mpDraw = mp.solutions.drawing_utils

'''
Function to process the data returned by mediapipe

Parameters:
    img - Image to be processed

Return:
    lmList - List of points on the palm
'''
def findHandPoints(img, handNo=0, draw=False):

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    
    if results.multi_hand_landmarks:
        # print(results.multi_handedness)
        myHand = results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark):
            # print(id, lm)
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            # print(id, cx, cy)
            lmList.append([id, cx, cy])
            if draw:
                # if id == 8 and (lmList[8][2] < lmList[6][2]) :
                #     self.line.append([cx, cy])
                mpDraw.draw_landmarks(img, myHand,
                                               mpHands.HAND_CONNECTIONS)
    return lmList

'''
Function to make a list with the 0's and 1's corresponding the number of fingers raised

Parameters:
    fingerPoints - List of points retunred by findHandPoints()

Return:
    total - List of 0's and 1's
'''
def findTheNumberSet(fingerPoints):
    total = []
    if fingerPoints[tipIds[0]][1] < fingerPoints[tipIds[0]-1][1]:
        total.append(1)
    else:
        total.append(0)
    for i in range(1,5):
        if fingerPoints[tipIds[i]][2] < fingerPoints[tipIds[i]-2][2]:
            total.append(1)
        else:
            total.append(0)
    return total


# pTime = 0
# cTime = 0
tipIds = [4, 8, 12, 16, 20]


'''
Driver function
'''
def main():

    while True:
        success, img = cap.read()
        # cTime = time.time()
        # fps = 1 / (cTime - pTime)
        # pTime = cTime
        img = cv2.flip(img, 1)
        # Calling function for finding the hand points
        fingerPoints = findHandPoints(img)
        total = []
        if len(fingerPoints) != 0:

            # Calling function to get the number fingers that are raised
            total = findTheNumberSet(fingerPoints)

            # if fingerPoints[tipIds[1]][2] < fingerPoints[tipIds[1]-2][2]:
            #     cv2.circle(img, (fingerPoints[tipIds[1]][1], fingerPoints[tipIds[1]][2]), 15, (255, 0, 255), cv2.FILLED)

        # The number shown in the hand
        totalCount = total.count(1)

        # Draw the number on the camera feed in rectangle
        #cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, 'The Number', (20, 235), cv2.FONT_HERSHEY_PLAIN,
            2, (255, 0, 0), 2)
        cv2.putText(img, str(totalCount), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 15)

        # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
        #             (255, 0, 255), 3)

        cv2.imshow("Feed", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()