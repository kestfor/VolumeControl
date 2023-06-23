import cv2
import time
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import HandTrackingModule as htm
import keyboard
import sys


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
MIN_VOLUME, MAX_VOLUME, a = volume.GetVolumeRange()

WIDTH = 640
HEIGHT = 480
if len(sys.argv) == 5:
    first_name, second_name = sys.argv[1], sys.argv[3]
    if first_name == '-w' and second_name == '-h':
        WIDTH = int(sys.argv[2])
        HEIGHT = int(sys.argv[4])
elif len(sys.argv) != 1:
    raise SyntaxError("необязательные аргументы: -w, -h")


STANDARD_BOX_HEIGHT = 285
STANDARD_BOX_WIDTH = 280
STANDARD_BOX_SIZE = (STANDARD_BOX_WIDTH, STANDARD_BOX_HEIGHT)
STANDARD_MAX_DISTANCE = 230
STANDARD_MIN_DISTANCE = 25
STANDARD_FINGER_LEN = 41
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
curr_time = 0
prev_time = 0
area = 0
detector = htm.HandDetector(max_num_hands=1, min_detection_confidence=0.9, min_tracking_confidence=0.9)
curr_volume = np.interp(volume.GetMasterVolumeLevel(), (MIN_VOLUME, MAX_VOLUME), (0, 100))

try_prev = False
try_prev_counter = 0
counter = 0
enable_fps = False
enable_draw = False
text_type = 0
timer = 0


def get_text(type_of_text):
    if type_of_text == 1:
        return "Next Track"
    if type_of_text == 2:
        return "Previous Track"
    if type_of_text == 3:
        return "Play/Pause"


while True:

    success, img = cap.read()
    img = cv2.flip(img, 1)

    detector.set_hands(img, draw=enable_draw)
    landmarks, hand_box = detector.find_position(img, draw=enable_draw)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    if enable_fps:
        cv2.putText(img, str(int(fps)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, detector.black_color, 3)

    if len(landmarks) > 0:
        box_width = abs(hand_box[0] - hand_box[2])
        box_height = abs(hand_box[1] - hand_box[3])
        area = box_height * box_width // 100
        box_size = (box_width, box_height)

        if 0 <= area <= 1700:
            fingers = detector.fingers_up()
            gest1 = detector.find_distance(img, 4, 14, STANDARD_BOX_SIZE, box_size)[0]
            gest2 = detector.find_distance(img, 4, 15, STANDARD_BOX_SIZE, box_size)[0]
            if fingers[0] and not (gest1 <= 60 and gest2 <= 60):
                curr_finger_len = detector.find_distance(img, 6, 7)[0]
                distance, circle = detector.find_distance(img, 4, 8, STANDARD_FINGER_LEN, curr_finger_len, draw=True)
                cv2.circle(img, circle, 10, detector.white_color, cv2.FILLED)
                volume_percentage = np.interp(distance, (STANDARD_MIN_DISTANCE, STANDARD_MAX_DISTANCE), (0, 100))
                smoothness = 5
                volume_percentage = smoothness * int((volume_percentage / smoothness))
                if all(fingers[2:]):
                    volume.SetMasterVolumeLevelScalar(volume_percentage / 100, None)
                    if curr_volume != volume_percentage:
                        curr_volume = volume_percentage
                    cv2.circle(img, circle, 10, detector.green_color, cv2.FILLED)
            elif fingers[1] and fingers[2] and not any(fingers[3:]):
                curr_finger_len = detector.find_distance(img, 6, 7)[0]
                distance, circle = detector.find_distance(img, 8, 12, STANDARD_FINGER_LEN, curr_finger_len, draw=True)
                cv2.circle(img, circle, 10, detector.white_color, cv2.FILLED)
                if distance <= 60:
                    counter += 1
                    cv2.circle(img, circle, 25, detector.green_color, cv2.FILLED)

                if distance > 110:
                    if try_prev:
                        try_prev_counter += 1
                    if try_prev_counter > fps // 2:
                        cv2.circle(img, circle, 10, detector.green_color, cv2.FILLED)
                        print("Next Track")
                        keyboard.send("next track")
                        timer = 0
                        text_type = 1
                        try_prev_counter = 0
                        try_prev = False
                    if counter >= fps // 2:
                        cv2.circle(img, circle, 10, detector.green_color, cv2.FILLED)
                        print("Play/Pause")
                        keyboard.send("play/pause")
                        timer = 0
                        text_type = 3
                        try_prev = False
                        try_prev_counter = 0
                    elif counter > fps // 8:
                        if not try_prev:
                            try_prev = True
                            try_prev_counter = 0
                        else:
                            if try_prev_counter <= fps // 3:
                                cv2.circle(img, circle, 10, detector.green_color, cv2.FILLED)
                                print("Previous Track")
                                keyboard.send("previous track")
                                timer = 0
                                text_type = 2
                                try_prev_counter = 0
                                try_prev = False
                    counter = 0

    if timer == 0 and text_type > 0:
        timer = time.time() + 1.5

    if timer > time.time():
        cv2.putText(img, get_text(text_type), (0, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, detector.black_color, 6)
    else:
        timer = 0
        text_type = 0

    cv2.imshow("IMG", img)
    action = cv2.waitKey(1)

    if action == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

    if action == ord('f'):
        enable_fps = not enable_fps

    if action == ord('d'):
        enable_draw = not enable_draw
