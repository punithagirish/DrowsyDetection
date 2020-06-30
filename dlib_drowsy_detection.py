# python dlib_drowsy_detection.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
import cv2
import argparse
import dlib
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread
import playsound

# eye aspect ratio to indicate blink
EYE_AR_THRESH = 0.3
# number of consecutive frames the eye must be below the threshold for to set off the alarm
EYE_AR_CONSEC_FRAMES = 48

# initialize the frame counter
# indicate if the alarm is going off
COUNTER = 0
DROWSY_ALARM_ON = False


def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)


def facial_landmark_coordinates(landmarks, dtype="int"):
    # return the list of (x, y)-coordinates
    return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, landmarks.num_parts)])


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
                help="path alarm .WAV file")

args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

print("[INFO] starting video stream thread...")
vc = cv2.VideoCapture(0)
while True:
    ret, frame = vc.read()
    if ret:
        frame = cv2.resize(frame, (450, 450))
        cv2.imshow('frame', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        faces = detector(gray)

        # loop over the face detections
        for face in faces:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            landmarks = predictor(gray, face)
            shape = facial_landmark_coordinates(landmarks)
            (lStart, lEnd) = (42, 48)
            (rStart, rEnd) = (36, 42)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1

                # if the eyes were closed for a sufficient number of
                # then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    # if the alarm is not on, turn it on
                    if not DROWSY_ALARM_ON:
                        DROWSY_ALARM_ON = True

                        # check to see if an alarm file was supplied,
                        # and if so, start a thread to have the alarm
                        # sound played in the background
                        if args["alarm"] != "":
                            t = Thread(target=sound_alarm,
                                       args=(args["alarm"],))
                            t.deamon = True
                            t.start()

                    # draw an alarm on the frame
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
            else:
                COUNTER = 0
                DROWSY_ALARM_ON = False

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

vc.release()
cv2.destroyAllWindows()
