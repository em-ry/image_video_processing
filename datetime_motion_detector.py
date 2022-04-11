import cv2,pandas
from datetime import datetime

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#takes num or string as argument. num for cam and string for video file
video = cv2.VideoCapture(0,cv2.CAP_DSHOW)#changes the backend to DirectShow

first_frame = None
status_list = [None,None]
time_stamp = [] 
df = pandas.DataFrame(columns=["start","end"])

# By iterating through the frame display code, a real time video is produced based on the speed of the cv2.waitKey()
while True:
    #check is a boolean value and the frame captured by the video object is read into the variable 'frame'
    check, frame= video.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #blurring the image removes noise and increases accuracy
    gray = cv2.GaussianBlur(gray,(21,21),0)
    status = 0

    if first_frame is None:
        first_frame = gray
        continue #this tells the interpreter to ignore the remaining lines of code and run the iteration again

    delta_frame = cv2.absdiff(first_frame,gray)
    thresh_frame = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]#[1] is needed as result is a tuple of 2 items and the second one is the frame I need
    thresh_frame + cv2.dilate(thresh_frame,None,iterations=2)#to remove the black holes in the bigger white images

    (cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        status = 1
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    status_list.append(status)
    if status_list[-1] == 1 and status_list[-2] == 0:
        time_stamp.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        time_stamp.append(datetime.now())
    
    cv2.imshow("frame",frame)
    cv2.imshow("gray frame",gray)
    cv2.imshow("delta frame", delta_frame)
    cv2.imshow("threshold frame", thresh_frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        if status == 1:
            time_stamp.append(datetime.now())
        break

print(status_list)
print(time_stamp)

for i in range(0,len(time_stamp),2):
    #The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    df= df.append({"start": time_stamp[i],"end": time_stamp[i+1]},ignore_index=True)#Can only append a dict if ignore_index=True
df.to_csv("time_stamp.csv")

video.release()
cv2.destroyAllWindows()