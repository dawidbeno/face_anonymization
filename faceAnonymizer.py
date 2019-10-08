import sys, getopt
import numpy as np
import tensorflow as tf
import cv2
import time

def printHelp():
    print('faceAnonymizer.py -i<inputfile> -s')
    print("-i <inputfile>")
    print("\tfile to be anonymized")
    print("-s")
    print("\tselfie mode")

def makeBoxGreatAgain(boxOld, imH, imW):
    boxOld[0] *= imH
    boxOld[1] *= imW
    boxOld[2] *= imH
    boxOld[3] *= imW
    box = []
    for b in boxOld:
        box.append(int(b))

    return box


class DetectionAPI:

# V tomto projekte sme pouzili kod, ktory sa nachadza v tutoriali ku kniznici Tensorflow
# na URL: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/camera.html
#
# Pouzili sme niekolko riadkov z tutorialu a vyskladali sme nasledujuce metody __init__() a processFrame()
# Casti prevzateho kodu sa tykaju hlavne inicializacie API detektoru objetov
#
# Zaciatok prevzateho kodu
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Extract image tensor
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Extract detection boxes
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Extract detection scores
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        # Extract detection classes
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        # Extract number of detectionsd
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

# Koniec prevzateho kodu

        print("Elapsed Time:", end_time-start_time)

        return boxes, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


def anonymousSociety(inputVideo, selfieMode):
    body_model_path = "faster_rcnn_inception_v2_coco/frozen_inference_graph.pb"
    face_model_path = "facessd_mobilenet_v2/frozen_inference_graph_face.pb"

    profileface_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_profileface.xml')
    rightEar_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_mcs_rightear.xml")
    leftEar_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_mcs_leftear.xml")

    bodyAPI = DetectionAPI(path_to_ckpt=body_model_path)
    faceAPI = DetectionAPI(path_to_ckpt=face_model_path)
    bodyThreshold = 0.7
    faceThreshold = 0.8

    cap = cv2.VideoCapture(str(inputVideo))

    videoLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the width and height of frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print "Video ma %d snimkov\n" % videoLength
    print "Video fps: %d\n" % fps
    print ("Rozlisenie: " + str(width) + " x " + str(height) + "\n")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use the lower case
    out = cv2.VideoWriter('ANONYMIZED_video.mp4', fourcc, fps, (width, height))

    frame_counter = 1
    foundFace = 0
    framesNum = 60

    while (cap.isOpened()):
        frames = []
        if frame_counter >= videoLength:
            break

        for i in range(framesNum):
            ret, frame = cap.read()
            frame_counter += 1
            print "Frame: %d\n" % frame_counter
            if ret == True:
                frames.append(frame)
            else:
                break
            if frame_counter >= videoLength:
                break

        print("Nacitane framy\n")

        if selfieMode == 0:
            for frame in frames:
                # Search for bodies
                bodyBoxes, bodyScores, bodyClasses, bodyNum = bodyAPI.processFrame(frame)
                for i in range(bodyBoxes.shape[1]):
                    # Class 1 represents human
                    if bodyClasses[i] == 1 and bodyScores[i] > bodyThreshold:
                        bodyBox = makeBoxGreatAgain(bodyBoxes[0, i], height, width)
                        bX = bodyBox[1]
                        bY = bodyBox[0]
                        bXW = bodyBox[3]
                        bYH = bodyBox[2]
                        sub_body = frame[bY:bYH, bX:bXW]

                        # faces in bodies
                        faceBoxes, faceScores, faceClasses, faceNum = faceAPI.processFrame(sub_body)
                        for j in range(len(faceBoxes)):
                            if faceClasses[j] == 1 and faceScores[j] > faceThreshold:
                                foundFace = 1
                                faceBox = makeBoxGreatAgain(faceBoxes[0, j], height, width)
                                fX = faceBox[1]
                                fY = faceBox[0]
                                fXW = faceBox[3]
                                fYH = faceBox[2]

                                sub_face = sub_body[fY:fYH, fX:fXW]
                                sub_face = cv2.GaussianBlur(sub_face, (33, 33), 30)
                                sub_body[fY:fY + sub_face.shape[0], fX:fX + sub_face.shape[1]] = sub_face
                                frame[bY:bY + sub_body.shape[0], bX:bX + sub_body.shape[1]] = sub_body
                                # cv2.rectangle(sub_body, (fX, fY), (fXW, fYH), (0, 0, 255), 2)

                        # blur upper body if face not found
                        if foundFace == 0:
                            if (bYH / 2) > bY:
                                bYH /= 2
                            sub_body = frame[bY:bYH, bX:bXW]
                            sub_body = cv2.GaussianBlur(sub_body, (33, 33), 30)
                            frame[bY:bY + sub_body.shape[0], bX:bX + sub_body.shape[1]] = sub_body
                            # cv2.rectangle(frame, (bX, bY), (bXW, bYH), (255, 0, 0), 2)

                        foundFace = 0

        # Tensorflow face
        for frame in frames:
            boxes, scores, classes, num = faceAPI.processFrame(frame)
            for i in range(len(boxes)):
                # Class 1 represents human
                if classes[i] == 1 and scores[i] > faceThreshold:
                    box = makeBoxGreatAgain(boxes[0, i], height, width)
                    x = box[1]
                    y = box[0]
                    xw = box[3]
                    yh = box[2]

                    sub_body = frame[y:yh, x:xw]
                    sub_body = cv2.GaussianBlur(sub_body, (33, 33), 30)
                    frame[y:y + sub_body.shape[0], x:x + sub_body.shape[1]] = sub_body
                    cv2.rectangle(frame, (x, y), (xw, yh), (0, 0, 255), 2)

        print ("Koniec tensorflow\n")
        # Cascades

        # profile face
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = profileface_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                flags=cv2.CASCADE_SCALE_IMAGE)

            if len(faces) > 0:
                # Draw a rectangle around the faces
                for fr in frames:
                    for (x, y, w, h) in faces:
                        sub_profileFace = fr[y:y + h, x:x + w]
                        sub_profileFace = cv2.GaussianBlur(sub_profileFace, (33, 33), 30)
                        fr[y:y + sub_profileFace.shape[0], x:x + sub_profileFace.shape[1]] = sub_profileFace
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                break
        print ("Koniec profile face\n")

        # rightEar
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rightEars = rightEar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                flags=cv2.CASCADE_SCALE_IMAGE)

            if len(rightEars) > 0:
                for fr in frames:
                    for (x, y, w, h) in rightEars:
                        x = x - w * 4
                        y = y - h
                        w *= 6
                        h *= 4
                        sub_ear = fr[y:y + h, x:x + w]
                        sub_ear = cv2.GaussianBlur(sub_ear, (33, 33), 30)
                        fr[y:y + h, x:x + w] = sub_ear
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                break
        print("Koniec rightEar\n")

        # leftEar
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            leftEars = leftEar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                flags=cv2.CASCADE_SCALE_IMAGE)

            if len(leftEars) > 0:
                for fr in frames:
                    for (x, y, w, h) in leftEars:
                        x = x - w / 2
                        y = y - h
                        w *= 6
                        h *= 4
                        sub_ear = fr[y:y + h, x:x + w]
                        sub_ear = cv2.GaussianBlur(sub_ear, (33, 33), 30)
                        fr[y:y + h, x:x + w] = sub_ear
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                break
        print ("Koniec leftEar\n")

        for frame in frames:
            out.write(frame)
            # cv2.imshow("video", frame)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        if frame_counter >= videoLength:
            break

    faceAPI.close()
    bodyAPI.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main(argv):
    inputfile = ''
    selfieMode = 0
    if len(argv) == 0:
        print("No file to be anonymized\n")
        printHelp()
        sys.exit(2)
    try:
        opts, args = getopt.getopt(argv, "hi:s")
    except getopt.GetoptError:
        print('faceAnonymizer.py -i <inputfile> -s')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            printHelp()
            sys.exit()
        elif opt in ("-i"):
            inputfile = arg
        elif opt in ("-s"):
            selfieMode = 1
            print("Selfie mode ON ...")
    print('File '+str(inputfile)+" will be anonymized ...\n")
    anonymousSociety(inputfile, selfieMode)





if __name__ == "__main__":
    main(sys.argv[1:])
