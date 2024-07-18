import cv2
import numpy as np
import time

#The script loads the YOLO model with weights and configuration files. 
net = cv2.dnn.readNet("./yolov3.weights","./yolov3.cfg") # Original yolov3
#net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg") #Tiny Yolo
classes = []
with open("coco.names","r") as f:  #It also loads the class names from a file named "coco.names."
    classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()  #Retrieves the names of all layers in the neural network.
    outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] # Identifies the output layers where object detections will be made.
    colors= np.random.uniform(0,255,size=(len(classes),3)) #Generates random colors for each class.
    #loading image
#cap=cv2.VideoCapture("22.mp4") #0 for 1st webcam
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_PLAIN
    starting_time= time.time()
    frame_id = 0

#The script enters a loop to continuously process frames from the camera.
    while True:
        _,frame= cap.read() 
        frame_id+=1
    
        height,width,channels = frame.shape
    #Each frame is preprocessed using the YOLO model's requirements. The image is resized to 320x320 pixels, and a blob is created.
        blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False) #reduce 416 to 320    

        
        net.setInput(blob)   #The preprocessed image is passed to the YOLO model using net.setInput(blob)
        outs = net.forward(outputlayers) #The model's forward pass is performed, and the results are obtained from the output layers.
    #print(outs[1])

    #The script iterates through the detected objects and applies confidence thresholds to filter out low-confidence detections.

    #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
        class_ids=[]
        confidences=[]
        boxes=[]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3: 
                #onject detected
                    center_x= int(detection[0]*width)
                    center_y= int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                #rectangle co-ordinaters
                    x=int(center_x - w/2)
                    y=int(center_y - h/2)
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                    boxes.append([x,y,w,h]) #put all rectangle areas
                    confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                    class_ids.append(class_id) #name of the object tha was detected

        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6) #Non-maximum suppression (cv2.dnn.NMSBoxes) is applied to remove duplicate and low-confidence boxes.


        for i in range(len(boxes)):  #For each detected object, a bounding box is drawn around it on the frame.
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence= confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2) 
                #The class label and confidence score are displayed near the bounding box.
            

        elapsed_time = time.time() - starting_time
        fps=frame_id/elapsed_time
        cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1) #The script calculates and displays the frames per second (FPS) on the video feed.
    
        cv2.imshow("Image",frame) #The processed frame is displayed in a window using cv2.imshow.
        key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
    
        if key == 27: #esc key stops the process
            break;
    
    cap.release()    
    cv2.destroyAllWindows()    
    #After exiting the loop, the video capture is released (cap.release()), and all OpenCV windows are closed (cv2.destroyAllWindows()).
    
    