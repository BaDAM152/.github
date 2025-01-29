import cv2
import os
import numpy as np
from google.colab.patches import cv2_imshow

class FaceRecognizer:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.persons = {}
        
    def prepare_training_data(self, training_dir):
        faces = []
        labels = []
        label_counter = 0
        
        for person_name in os.listdir(training_dir):
            person_dir = os.path.join(training_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            self.persons[label_counter] = person_name
            
            for image_file in os.listdir(person_dir):
                if image_file.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_dir, image_file)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    detected_faces = self.face_cascade.detectMultiScale(
                        image,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )
                    
                    for (x, y, w, h) in detected_faces:
                        face_roi = image[y:y+h, x:x+w]
                        faces.append(cv2.resize(face_roi, (200, 200)))
                        labels.append(label_counter)
            
            label_counter += 1
            
        return faces, labels
    
    def train(self, training_dir):
        print("Training started...")
        faces, labels = self.prepare_training_data(training_dir)
        
        if not faces:
            raise ValueError("No faces found in training data!")
            
        self.recognizer.train(faces, np.array(labels))
        print("Training completed!")
    
    def predict(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            label, confidence = self.recognizer.predict(face_roi)
            
            name = self.persons.get(label, "Unknown")
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, f"{name}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return image

def main():
    training_dir = "" # training directory
    test_image_path = "" # test 
    
    recognizer = FaceRecognizer()
    
    try:
        recognizer.train(training_dir)
        result_image = recognizer.predict(test_image_path)
        
        cv2.imwrite("result.jpg", result_image)
        cv2_imshow(result_image)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()