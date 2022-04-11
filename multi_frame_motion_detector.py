import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#this creates a video object which triggers the system's camera
#takes num or string as argument. num for cam and string for video file
video = cv2.VideoCapture(0,cv2.CAP_DSHOW)#changes the backend to DirectShow

first_frame = None 

# By iterating through the frame display code, a real time video is produced based on the speed of the cv2.waitKey()
while True:
    #check is a boolean value and the frame captured by the video object is read into the variable 'frame'
    check, frame= video.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #blurring the image removes noise and increases accuracy
    gray = cv2.GaussianBlur(gray,(21,21),0)

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

        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    cv2.imshow("frame",frame)
    cv2.imshow("gray frame",gray)
    cv2.imshow("delta frame", delta_frame)
    cv2.imshow("threshold frame", thresh_frame)

    key = cv2.waitKey(1)
    print(gray)
    print(delta_frame)

    if key == ord("q"):
        break



video.release()
cv2.destroyAllWindows()