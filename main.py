import cv2
from datetime import datetime
import pandas as pandas

first_frame = None
status_list = [0]
times = []
video = cv2.VideoCapture(0)
df = pandas.DataFrame(columns=["Entered", "Exited"])

while True:
    check, frame = video.read()
    status = 0 #no object in frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # image in gery scale
    gray = cv2.GaussianBlur(gray, (21,21), 0) # blur the image to cancel white noises

    if first_frame is None: # first frame is the baseline to compare with
        first_frame = gray
        continue
    # the delta frame that represents the differences between the first and the current image
    delta_frame = cv2.absdiff(first_frame, gray)

    # thresh the delta frame to see the difference more clearly
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # smooth the thresh frame - remove black holes in the white areas
    thresh_frame = cv2.dilate(thresh_delta, None, iterations=2)

    # find all the contours in the image
    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000: # object smaller then 100X100 pixels dont counts
            continue
        status = 1 # object in the frame

        (x, y, w, h) = cv2.boundingRect(contour) # draw a green rectangle around the objects
        cv2.rectangle(frame, pt1=(x,y), pt2=(x+w, y+h), color=(0,255,0), thickness=3)

    status_list.append(status)

    if status_list[-1] == 1 and status_list[-2] == 0: # if new object entered the frame
        times.append(datetime.now())

    if status_list[-1] == 0 and status_list[-2] == 1: # if the object exited the frame
        times.append(datetime.now())

    #cv2.imshow("press q to exit - gray", gray)
    #cv2.imshow("press q to exit - delta_frame", delta_frame)
    #cv2.imshow("press q to exit - thresh_delta", thresh_delta)
    #cv2.imshow("press q to exit - thresh_frame", thresh_frame)
    cv2.imshow("press q to exit - color frame", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        if status == 1: # if the object was in the frame when the program exited
            times.append(datetime.now())
        break

# adding the times that objects entered and exited to the frame to a csv file
for i in range(0, len(times), 2):
    df = df.append({"Entered": times[i], "Exited": times[i+1]}, ignore_index=True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows()
