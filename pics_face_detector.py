import cv2

# Haarcascade is a machine learning algorithm which is used to identify objects in images or videos
# creating a cascade classifier object
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

image = cv2.imread("files/photo.jpg")
#we convert the above image to gray format below. to be used for image search.
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# It will search for a cascade classifier and return coordinates of the face in the image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor= 1.09, minNeighbors = 10)
print(type(faces))
print(faces)


for x, y, w, h in faces:
    # cv2.rectangle(or any shape) displays a shape on a picture or video with coordinates given.
    image = cv2.rectangle(gray_image,(x,y),(x+w,y+h),(0,0,255),4)

img = cv2.resize(image,(int(gray_image.shape[1]/2),int(gray_image.shape[0]/2)))
cv2.imshow("flffl",img)
cv2.waitKey(0)
cv2.destroyAllWindows()