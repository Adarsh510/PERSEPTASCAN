import tkinter as tk
from tkinter import ttk
import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PerceptaScan")

        self.label1 = ttk.Label(root, text="PerceptaScan", foreground="light blue", font="Tahoma 20 bold")
        self.label1.pack()

        self.video_source = 0  # Use the default camera (change if using an external camera)
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(root, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.style = ttk.Style()
        self.style.map("TButton", background=[("active", "Red")])

        self.btn_start_stop = ttk.Button(root, text="Scan", command=self.toggle_recognition)
        self.btn_start_stop.pack(pady=10)

        self.exit_button = tk.Button(root, text="Exit", command=self.Close)
        self.exit_button.pack(pady=20)

        # Initialize face recognition variables
        self.face_names = []
        self.process_this_frame = True
        self.face_locations = []

        # Load face recognition models
        self.load_face_recognition_models()

        # Initialize object detection variables
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Replace with your YOLO model files
        self.classes = []
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.recognizing = False
        self.update()
        self.root.mainloop()

    def Close(self):
        self.root.destroy()

    def load_face_recognition_models(self):
        # Load face recognition models
        adarsh_image = face_recognition.load_image_file("Faces/Adarsh/Adarsh.jpg")
        adarsh_face_encoding = face_recognition.face_encodings(adarsh_image)[0]
        self.known_face_encodings = [adarsh_face_encoding]
        self.known_face_names = ["Adarsh"]

        reshma_image = face_recognition.load_image_file("Faces/Reshma/Reshma.jpg")
        reshma_face_encoding = face_recognition.face_encodings(reshma_image)[0]
        self.known_face_encodings.append(reshma_face_encoding)
        self.known_face_names.append("Reshma")

        Ashly_image = face_recognition.load_image_file("Faces/Ashly/Ashly.jpg")
        Ashly_face_encoding = face_recognition.face_encodings(Ashly_image)[0]
        self.known_face_encodings.append(Ashly_face_encoding)
        self.known_face_names.append("Ashly")

        Farhan_image = face_recognition.load_image_file("Faces/Farhan/Farhan.jpg")
        Farhan_face_encoding = face_recognition.face_encodings(Farhan_image)[0]
        self.known_face_encodings.append(Farhan_face_encoding)
        self.known_face_names.append("Farhan")

        Athul_image = face_recognition.load_image_file("Faces/Athul/Athul.jpg")
        Athul_face_encoding = face_recognition.face_encodings(Athul_image)[0]
        self.known_face_encodings.append(Athul_face_encoding)
        self.known_face_names.append("Athul")

        Jessica_image = face_recognition.load_image_file("Faces/Jessica/Jessica.jpg")
        Jessica_face_encoding = face_recognition.face_encodings(Jessica_image)[0]
        self.known_face_encodings.append(Jessica_face_encoding)
        self.known_face_names.append("Jessica")

        Sebastian_image = face_recognition.load_image_file("Faces/Sebastian/Sebastian.jpg")
        Sebastian_face_encoding = face_recognition.face_encodings(Sebastian_image)[0]
        self.known_face_encodings.append(Sebastian_face_encoding)
        self.known_face_names.append("Sebastian")

        Vyshnavi_image = face_recognition.load_image_file("Faces/Vyshnavi/Vyshnavi.jpg")
        Vyshnavi_face_encoding = face_recognition.face_encodings(Vyshnavi_image)[0]
        self.known_face_encodings.append(Vyshnavi_face_encoding)
        self.known_face_names.append("Vyshnavi")

        Akhil_image = face_recognition.load_image_file("Faces/Akhil/Akhil.jpg")
        Akhil_face_encoding = face_recognition.face_encodings(Akhil_image)[0]
        self.known_face_encodings.append(Akhil_face_encoding)
        self.known_face_names.append("Akhil")

        Sreerag_image = face_recognition.load_image_file("Faces/Sreerag/Sreerag.jpg")
        Sreerag_face_encoding = face_recognition.face_encodings(Sreerag_image)[0]
        self.known_face_encodings.append(Sreerag_face_encoding)
        self.known_face_names.append("Sreerag")

        Aiswarya_image = face_recognition.load_image_file("Faces/Aiswarya/Aiswarya.jpg")
        Aiswarya_face_encoding = face_recognition.face_encodings(Aiswarya_image)[0]
        self.known_face_encodings.append(Aiswarya_face_encoding)
        self.known_face_names.append("Aiswarya")

        Sini_image = face_recognition.load_image_file("Faces/Sini/Sini.jpg")
        Sini_face_encoding = face_recognition.face_encodings(Sini_image)[0]
        self.known_face_encodings.append(Sini_face_encoding)
        self.known_face_names.append("Sini")

        Dhanush_image = face_recognition.load_image_file("Faces/Dhanush/Dhanush.jpg")
        Dhanush_face_encoding = face_recognition.face_encodings(Dhanush_image)[0]
        self.known_face_encodings.append(Dhanush_face_encoding)
        self.known_face_names.append("Dhanush")

        Jouhar_image = face_recognition.load_image_file("Faces/Jouhar/Jouhar.jpg")
        Jouhar_face_encoding = face_recognition.face_encodings(Jouhar_image)[0]
        self.known_face_encodings.append(Jouhar_face_encoding)
        self.known_face_names.append("Jouhar")

        Sharon_image = face_recognition.load_image_file("Faces/Sharon/Sharon.jpg")
        Sharon_face_encoding = face_recognition.face_encodings(Sharon_image)[0]
        self.known_face_encodings.append(Sharon_face_encoding)
        self.known_face_names.append("Sharon")

        Remil_image = face_recognition.load_image_file("Faces/Remil/Remil.jpg")
        Remil_face_encoding = face_recognition.face_encodings(Remil_image)[0]
        self.known_face_encodings.append(Remil_face_encoding)
        self.known_face_names.append("Remil")

        Navneeth_image = face_recognition.load_image_file("Faces/Navneeth/Navneeth.jpg")
        Navneeth_face_encoding = face_recognition.face_encodings(Navneeth_image)[0]
        self.known_face_encodings.append(Navneeth_face_encoding)
        self.known_face_names.append("Navneeth")

        Abin_image = face_recognition.load_image_file("Faces/Abin/Abin.jpg")
        Abin_face_encoding = face_recognition.face_encodings(Abin_image)[0]
        self.known_face_encodings.append(Abin_face_encoding)
        self.known_face_names.append("Abin")

        Tina_image = face_recognition.load_image_file("Faces/Tina/Tina.jpg")
        Tina_face_encoding = face_recognition.face_encodings(Tina_image)[0]
        self.known_face_encodings.append(Tina_face_encoding)
        self.known_face_names.append("Tina")

        Arjun_image = face_recognition.load_image_file("Faces/Arjun/Arjun.jpg")
        Arjun_face_encoding = face_recognition.face_encodings(Arjun_image)[0]
        self.known_face_encodings.append(Arjun_face_encoding)
        self.known_face_names.append("Arjun")

        Anand_S_image = face_recognition.load_image_file("Faces/Anand_S/Anand_S.jpg")
        Anand_S_face_encoding = face_recognition.face_encodings(Anand_S_image)[0]
        self.known_face_encodings.append(Anand_S_face_encoding)
        self.known_face_names.append("Anand_S")

        Hari_image = face_recognition.load_image_file("Faces/Hari/Hari.jpg")
        Hari_face_encoding = face_recognition.face_encodings(Hari_image)[0]
        self.known_face_encodings.append(Hari_face_encoding)
        self.known_face_names.append("Hari")

        Malavika_image = face_recognition.load_image_file("Faces/Malavika/Malavika.jpg")
        Malavika_face_encoding = face_recognition.face_encodings(Malavika_image)[0]
        self.known_face_encodings.append(Malavika_face_encoding)
        self.known_face_names.append("Malavika")

        Reshma_M_U_image = face_recognition.load_image_file("Faces/Reshma_M_U/Reshma_M_U.jpg")
        Reshma_M_U_face_encoding = face_recognition.face_encodings(Reshma_M_U_image)[0]
        self.known_face_encodings.append(Reshma_M_U_face_encoding)
        self.known_face_names.append("Reshma_M_U")

        Surag_image = face_recognition.load_image_file("Faces/Surag/Surag.jpg")
        Surag_face_encoding = face_recognition.face_encodings(Surag_image)[0]
        self.known_face_encodings.append(Surag_face_encoding)
        self.known_face_names.append("Surag")

        Aneena_image = face_recognition.load_image_file("Faces/Aneena/Aneena.jpg")
        Aneena_face_encoding = face_recognition.face_encodings(Aneena_image)[0]
        self.known_face_encodings.append(Aneena_face_encoding)
        self.known_face_names.append("Aneena")

        Anand_K_image = face_recognition.load_image_file("Faces/Anand_K/Anand_K.jpg")
        Anand_K_face_encoding = face_recognition.face_encodings(Anand_K_image)[0]
        self.known_face_encodings.append(Anand_K_face_encoding)
        self.known_face_names.append("Anand_K")

        Swathi_image = face_recognition.load_image_file("Faces/Swathi/Swathi.jpg")
        Swathi_face_encoding = face_recognition.face_encodings(Swathi_image)[0]
        self.known_face_encodings.append(Swathi_face_encoding)
        self.known_face_names.append("Swathi")

        # Aneena_image = face_recognition.load_image_file("Faces/Aneena/Aneena.jpg")
        # Aneena_face_encoding = face_recognition.face_encodings(Aneena_image)[0]
        # self.known_face_encodings.append(Aneena_face_encoding)
        # self.known_face_names.append("Aneena")
        #
        # Aneena_image = face_recognition.load_image_file("Faces/Aneena/Aneena.jpg")
        # Aneena_face_encoding = face_recognition.face_encodings(Aneena_image)[0]
        # self.known_face_encodings.append(Aneena_face_encoding)
        # self.known_face_names.append("Aneena")

    def toggle_recognition(self):
        if self.recognizing:
            self.recognizing = False
            self.btn_start_stop["text"] = "Scan"
        else:
            self.recognizing = True
            self.btn_start_stop["text"] = "Stop"

    def update(self):
        ret, frame = self.vid.read()

        if self.recognizing:
            self.face_names = self.perform_face_recognition(frame)
            self.perform_object_detection(frame)

        if ret:
            self.draw_on_canvas(frame)

        if self.recognizing:
            self.root.after(10, self.update)
        else:
            self.root.after(100, self.update)

    def perform_face_recognition(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if self.process_this_frame:
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                face_names.append(name)

            self.face_names = face_names

        self.process_this_frame = not self.process_this_frame
        return self.face_names

    def perform_object_detection(self, frame):
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = self.colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def draw_on_canvas(self, frame):
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.photo = photo

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
