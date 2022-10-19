import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folderPath = 'Header'
myList = os.listdir(folderPath)
overlayList = []
for imgPath in myList:
    sideBarImg = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(sideBarImg)
sideBar = overlayList[0]
drawingColor = (0, 0, 0)
brushThickness = 15
eraserThickness = 35

# Capturing video stream and setting the width to 1280 and the height to 720.
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
# Creating a Black img for masking.
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    # Mirroring the img.
    img = cv2.flip(img, 1)
    # Detecting and drawing the landmarks of the hand.
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # The positions of the tip of the index and middle fingers.
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # Check which fingers are up
        fingers = detector.fingersUp()

        # If the index and the middle fingers are up: selection mode.
        if fingers[0] == False and fingers[1] and fingers[2] and fingers[3] == False and fingers[4] == False:
            xp, yp = 0, 0
            print("Selection Mode")
            # Selecting the color based on the position of the tip of the two fingers.
            if x1 > 1156:
                if 30 < y1 < 90:
                    sideBar = overlayList[1]
                    drawingColor = (0, 0, 255)
                elif 150 < y1 < 220:
                    sideBar = overlayList[2]
                    drawingColor = (255, 0, 0)
                elif 260 < y1 < 340:
                    sideBar = overlayList[3]
                    drawingColor = (0, 255, 0)
                elif 390 < y1 < 460:
                    sideBar = overlayList[4]
                    drawingColor = (255, 0, 255)
                elif 510 < y1 < 600:
                    sideBar = overlayList[5]
                    drawingColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawingColor, cv2.FILLED)

        # If only the index finger is up: Drawing mode.
        elif fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 10, drawingColor, cv2.FILLED)
            print("Drawing Mode")
            # Drawing by making a line between two points - the current point and the last point.
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawingColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawingColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawingColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawingColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawingColor, brushThickness)

            xp, yp = x1, y1

        # If all the fingers are up: Eraser mode
        elif fingers[0] and fingers[1] and fingers[2] and fingers[3] and fingers[4]:
            print('Eraser Mode')
            x1, y1 = lmList[9][1:]
            cv2.circle(img, (x1, y1), 100, (0, 0, 0), cv2.FILLED)
            cv2.circle(imgCanvas, (x1, y1), 100, (0, 0, 0), cv2.FILLED)

        # If all the fingers are down: Stopping mode (won't draw/select/erase)
        elif not fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
            xp, yp = 0, 0
            print("Stopping Mode")

    # Masking
    # Creating a gray image.
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    # Converting the gray image to a binary image and invert it so 1 will be 0 and 0 will be 1
    # the background will be white and the drawing will be black no metter with color in selected.
    ret, imgInv = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV)
    # Converting back to BGR. (in order to add imgInv to the original img they have to be with the same dimensions)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    # Bitwise and operation.
    # The original colors will be painted in black.
    img = cv2.bitwise_and(img, imgInv)
    # Bitwise or operation.
    # The original colors will be painted with the selected color.
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the Toolbar img
    img[0:720, 1156:1280] = sideBar

    cv2.imshow("image", img)


    cv2.waitKey(1)
