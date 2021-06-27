import numpy as np
import cv2
import mediapipe as mp
import os

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

# Tip point index of each finger
tipIds = [4, 8, 12, 16, 20]

# Initialize the object for calling hand prediction from mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(mode, maxHands,detectionCon, trackCon)
mpDraw = mp.solutions.drawing_utils

# Initialize variables for editmode
inEditMode = False
drawList = []
icons = []
folderPath = "icons"
iconList = os.listdir(folderPath)
for iconPath in iconList:
    image = cv2.imread(f'{folderPath}/{iconPath}')
    icons.append(cv2.resize(image, (120, 120)))

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
        handCoor = results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(handCoor.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
            if draw:
                mpDraw.draw_landmarks(img, handCoor,
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

'''
Function to change display and edit mode
Parameters:
    point - The coordinated of the index finger
'''
def checkIfInEditMode(point):
    global inEditMode
    if point[1] < 120 and point[2] < 120:
        inEditMode = True

'''
Function to save the drawn points
Parameters:
    point - The coordinated of the index finger
'''
def markPoints(point):
    drawList.append([point[1], point[2]])

'''
Function to clear the drawn points
'''
def clearPoints(point):
    global drawList, inEditMode
    if point[2] > 120 and point[2] < 240 and point[1] < 120:
        drawList = []
        inEditMode = False

'''
Function to draw all the marked points
'''
def drawAllPoints(img):
    for point in drawList:
        cv2.circle(img, (point[0], point[1]), 15, (255, 0, 255), cv2.FILLED)

'''
Driver function
'''
def main():

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        # Calling function for finding the hand points
        fingerPoints = findHandPoints(img)
        total = []
        if len(fingerPoints) != 0:

            # Calling function to get the number fingers that are raised
            total = findTheNumberSet(fingerPoints)

            if fingerPoints[tipIds[1]][2] < fingerPoints[tipIds[1]-2][2]:

                # Function call to mark the points in edit mode
                if inEditMode:
                    markPoints(fingerPoints[tipIds[1]])
                else:
                    # Function call to check if going to edit mode
                    checkIfInEditMode(fingerPoints[tipIds[1]])

                #Function to clear all the drawing
                clearPoints(fingerPoints[tipIds[1]])

        # Function to draw all the points
        drawAllPoints(img)
        
        # The number shown in the hand
        totalCount = total.count(1)

        # Draw the number on the camera feed in rectangle
        cv2.putText(img, 'Number Shown', (380, 80), cv2.FONT_HERSHEY_PLAIN,
            2, (0, 0, 0), 2)
        cv2.putText(img, str(totalCount), (550, 150), cv2.FONT_HERSHEY_PLAIN,
                    5, (0, 0, 0), 10)

        # Drawing the Edit and Erase icons
        h1, w1, c1 = icons[1].shape
        img[0:h1, 0:w1] = icons[1]
        h2, w2, c2 = icons[0].shape
        img[h1:h1+h2, 0:w2] = icons[0]

        cv2.imshow("Feed", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()