from PIL import Image, ImageDraw
import face_recognition
import numpy as np

def find_landmarks(image, indx, show=False):
    face_landmarks_list = face_recognition.face_landmarks(image)

    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            d.line(face_landmarks[facial_feature], width=1)
    if show:
        pil_image.show(title=indx)

    return face_landmarks_list

    


def find_faces(iamge):
    face_locations = face_recognition.face_locations(image)

    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    faces = []

    pil_image = Image.fromarray(image)

    for indx, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]

        faces.append(face_image)

        draw = ImageDraw.Draw(pil_image)
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        text_width, text_height = draw.textsize(str(indx))
        draw.text((left + 6, bottom - text_height - 5), str(indx), fill=(255, 255, 255, 255))

    del draw
    print()    
    pil_image.show()

    return faces

if __name__ == "__main__":
    image = face_recognition.load_image_file("peps.jpg")

    faces = find_faces(image)

    max_dims = (0,0)
    for face in faces:
        dims = (face.shape[0], face.shape[1])
        if (dims[0] * dims[1]) > (max_dims[0] * max_dims[1]):
            max_dims = dims 

    face_data = []

    for indx, face in enumerate(faces, 0):
        face_data.append({})

        face = np.asarray(Image.fromarray(face).resize(max_dims))
        for key, val in find_landmarks(face, indx)[0].items():
            face_data[indx][key] = val
        
    key_set = face_data[0].keys()

    similarities = {}

    for i in range(0, len(face_data)):
        for j in range(i+1, len(face_data)):
            sim_key = str(i) + " to " + str(j)
            similarities[sim_key] = {}
            print(sim_key)
            scores = []
            for key in key_set:
                similarities[sim_key][key] = np.asarray(face_data[i][key]) - np.asarray(face_data[j][key])
                score = np.abs(similarities[sim_key][key].mean())
                scores.append(score)
                print(key + " has a difference value of " + str(score))

            print("Total difference: " + str(np.mean(scores)))
            print()
   