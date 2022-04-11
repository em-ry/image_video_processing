import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#this creates a video object which triggers the system's camera
#takes num or string as argument. num for cam and string for video file
video = cv2.VideoCapture(0,cv2.CAP_DSHOW)#changes the backend to DirectShow

# By iterating through the frame display code, a real time video is produced based on the speed of the cv2.waitKey()
a = 0 # we'll use this variable to count the frames
while True:
    a = a + 1
    #check is a boolean value and the frame captured by the video object is read into the variable 'frame'
    check, frame= video.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.05, minNeighbors= 5)#returns face coordinates

    for x, y, w, h in faces:
        gray = cv2.rectangle(gray,(x,y),(x+w,y+h),(200,50,100),4)
    cv2.imshow("any",gray)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break


print(a)
video.release()
cv2.destroyAllWindows()