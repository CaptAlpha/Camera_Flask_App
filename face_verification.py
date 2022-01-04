from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
import warnings

warnings.filterwarnings("ignore")

def create_bbox(image):
    detector=MTCNN()
    faces=detector.detect_faces(image)
    bounding_box=faces[0]['box']
    cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]),(0,155,255),2)
    return image

def extract_face(image,resize=(224,224)):
    detector=MTCNN()
    image=cv2.imread(image)
    faces=detector.detect_faces(image)
    x1,y1,width,height=faces[0]['box']
    x2,y2=x1+width,y1+height
    face_boundary=image[y1:y2,x1:x2]
    face_image=cv2.resize(face_boundary,resize)
    return face_image

def get_embeddings(faces):
    face=np.asarray(faces,'float32')
    face=preprocess_input(face, version=2)
    model=VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
    return model.predict(face)

def get_similarity(faces):
    embeddings=get_embeddings(faces)
    score=cosine(embeddings[0],embeddings[1])
    if score<=0.5:
        return "Face Matched",score
    return "Face Not Matched",score

def verify_face():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Face", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "face.png"
            cv2.imwrite(img_name, frame)
            print("Written!")
            faces=[extract_face(image) for image in ['static\Photo.jpg','face.png']]
            print(get_similarity(faces))

    cam.release()

    cv2.destroyAllWindows()

verify_face()