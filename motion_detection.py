"""
Author : Anuj Kulkarni
Topic : Multiple Moving Object Detection And Tracking.
Notes: Press escape button to end the program
"""
import cv2
import numpy as np

# Parsing The weights and configuration files
net = cv2.dnn.readNet('yolov3-35.weights', 'yolov3-35.cfg')

#Readin all the object names from the coco file to be detected
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Opening the camera video node and capturing the frames
cap = cv2.VideoCapture(0)
if not (cap.isOpened()):
    print('Could not open video device')

# Setting the frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#Reading the first frame and display settings
_, frame2 = cap.read()
font = cv2.FONT_HERSHEY_SIMPLEX
colors = np.random.uniform(0, 255, size=(100, 3))
imageframe = 1

while cap.isOpened():
    while imageframe % 5 == 0:
        _, frame1 = cap.read()
        height, width, _ = frame1.shape

        # Capturing the blob from the image frame with the weight of 1/255 and passing it to input to detect the elements
        blob = cv2.dnn.blobFromImage(frame1, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        # Calculates the confidence and supplies a class_id after referencing from the training set
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        # Displaying a box around the detected object and displaying the identified object
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(frame1, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame1, label + " " + confidence, (x, y + 20), font, 1, (255, 0, 0), 2)

        #Finding the difference between the first 2 frames in order to identify the new objects in the frams
        difference = cv2.absdiff(frame1, frame2)
        gray_image = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        smooth_image = cv2.GaussianBlur(gray_image, (5, 5), 3)
        _, thresh = cv2.threshold(smooth_image, 50, 255, cv2.THRESH_BINARY)
        dilate_image = cv2.dilate(thresh, None, iterations=3)
        #Using coutour tranform to detect the moving objects in the frame.
        contour_transform, _ = cv2.findContours(dilate_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Bounding the object detected from coutour transform with a box.
        for contour in contour_transform:
            (x, y, width, height) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) > 4500:
                cv2.rectangle(frame1, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame1, "Status: {}".format('Movement / Object Detected'), (10, 20), cv2.FONT_HERSHEY_PLAIN,
                            1.0, (0, 0, 255), 1)

        # Display the resulting frame from contour transform and the Threshold mask
        cv2.imshow('Treshold (foreground mask)', dilate_image)
        cv2.imshow('Image', frame1)

        frame1 = frame2
        _, frame2 = cap.read()

        # Waits for a user input to quit the application
        if cv2.waitKey(1) & 0xFF == 27:
            break
    imageframe += 1

cap.release()
cv2.destroyAllWindows()
