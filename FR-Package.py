import face_recognition
import os
import cv2

train = "Dataset/Training_set"
test = "Dataset/Testing_set"
tolerance = 0.6
frame_thickness = 3
font_thickness = 2
model = "cnn"

train_faces = []
train_names = []

for name in os.listdir(train):
    for filename in os.listdir(f"{train}/{name}"):
        image = face_recognition.load_image_file(f"{train}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        train_faces.append(encoding)
        train_names.append(name)

print("Processing test pictures...")

for filename in os.listdir(test):
    print(filename)
    image = face_recognition.load_image_file(f"{test}/{filename}")
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtCOLOR(image, cv2.COLOR_RGB2BGR)

for face_encoding, face_location in zip(encodings, locations):
    results = face_recognition.compare_faces(train_faces, face_encoding, tolerance)
    match = None
    if True in results:
        match = train_names[results.index(True)]
        print(f"Match found: {match}")
        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])

        color = [0,255,0]

        cv2.rectangle(image, top_left, bottom_right, color, frame_thickness)

        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2]+22)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), font_thickness)

    cv2.imshow(filename, image)
    cv2.waitKey(0)