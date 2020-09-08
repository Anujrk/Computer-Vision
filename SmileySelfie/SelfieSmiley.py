import cv2

cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_eye = cv2.CascadeClassifier('haarcascade_eye.xml')
cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')
count = 0


def detection(grayscale, image, count):
    face = cascade_face.detectMultiScale(grayscale, 1.3, 5)
    for (x_face, y_face, w_face, h_face) in face:
        # cv2.rectangle(image,(x_face,y_face),(x_face+w_face,y_face+h_face),(255,130,0),2)
        new_grayscale = grayscale[y_face:y_face + h_face, x_face:x_face + w_face]
        new_color = image[y_face:y_face + h_face, x_face:x_face + w_face]
        # eye = cascade_eye.detectMultiScale(new_grayscale, 1.2, 18)
        # for (x_eye, y_eye, w_eye, h_eye) in eye:
        #     cv2.rectangle(new_color, (x_eye, y_eye), (x_eye + w_eye, y_eye + h_eye), (0, 180, 60), 2)
        smile = cascade_smile.detectMultiScale(new_grayscale, 5.5, 20)
        if len(smile) > 0:
            count += 1
            if count % 10 == 0:
                cv2.imwrite("selfie%d.JPEG" % (count / 10), image)
        print(smile)
        for (x_smile, y_smile, w_smile, h_smile) in smile:
            cv2.rectangle(new_color, (x_smile, y_smile), (x_smile + w_smile, y_smile + h_smile), (255, 0, 130), 2)
    return image, count


frame = cv2.VideoCapture(0)

while True:
    _, image = frame.read()
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected, count = detection(grayscale, image, count)
    cv2.imshow('Smile Selfie', detected)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

frame.release()
cv2.destroyAllWindows()
