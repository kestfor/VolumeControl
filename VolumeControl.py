import cv2
import time
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import HandTrackingModule as htm
import keyboard
import sys


class Timer:
    def __init__(self):
        self._start = None
        self._end = None
        self._started = False
        self._ended = True

    def start(self):
        if self._ended:
            self._start = time.time()
            self._started = True
            self._ended = False

    def stop(self):
        if self._started:
            self._end = time.time()
            self._ended = True
            self._started = False

    def reset(self):
        self.__init__()

    def get_time(self):
        if self._start is not None:
            if self._end is None:
                return time.time() - self._start
            else:
                return self._end - self._start
        else:
            return 0


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


SETTING_TIME = 3
PAUSE_PLAY_TIME = 0.7
PREVIOUS_TRACK_TIME = 1
NEXT_TRACK_TIME = 0.1
STANDARD_BOX_HEIGHT = 285
STANDARD_BOX_WIDTH = 280
STANDARD_BOX_SIZE = [STANDARD_BOX_WIDTH, STANDARD_BOX_HEIGHT]
STANDARD_MAX_DISTANCE = 230
STANDARD_MIN_DISTANCE = 25
STANDARD_FINGER_LEN = 41
SECOND_GESTURE_MAX_LEN = 110

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
curr_time = 0
prev_time = 0
area = 0
detector = htm.HandDetector(max_num_hands=1, min_detection_confidence=0.9, min_tracking_confidence=0.9)
curr_volume = np.interp(volume.GetMasterVolumeLevel(), (MIN_VOLUME, MAX_VOLUME), (0, 100))

try_prev = False
try_prev_time = None
enable_fps = False
enable_draw = False
text_type = 0
timer = 0
gesture_timer = Timer()
setting_timer = Timer()
set_size_first = False
set_size_second = False

fingers_len_for_setting = []
max_dist_for_setting = []
box_size_for_setting = []
max_dist_second_gest_setting = []
times = set()


def set_size_first_gesture(finger_len, dist, hand_box, set_timer):
    global times
    curr_time = int(set_timer.get_time())
    fingers_len_for_setting.append(finger_len)
    max_dist_for_setting.append(dist)
    box_size_for_setting.append(hand_box)
    times.add(SETTING_TIME - curr_time)
    if curr_time >= SETTING_TIME:
        global STANDARD_FINGER_LEN
        global STANDARD_MAX_DISTANCE
        global STANDARD_BOX_SIZE
        global set_size_first
        set_size_first = False
        set_timer.reset()
        times.clear()
        STANDARD_FINGER_LEN = int(sum(fingers_len_for_setting) / len(fingers_len_for_setting))
        STANDARD_MAX_DISTANCE = int(sum(max_dist_for_setting) / len(max_dist_for_setting))
        STANDARD_BOX_SIZE[0] = int(sum([item[0] for item in box_size_for_setting]) / len(box_size_for_setting))
        STANDARD_BOX_SIZE[1] = int(sum([item[1] for item in box_size_for_setting]) / len(box_size_for_setting))
        fingers_len_for_setting.clear()
        max_dist_for_setting.clear()
        box_size_for_setting.clear()
        print(STANDARD_FINGER_LEN, STANDARD_MAX_DISTANCE, STANDARD_BOX_SIZE)


def set_size_second_gesture(dist, set_timer):
    global times
    curr_time = int(set_timer.get_time())
    max_dist_second_gest_setting.append(dist)
    times.add(SETTING_TIME - curr_time)
    if curr_time >= SETTING_TIME:
        global SECOND_GESTURE_MAX_LEN
        global set_size_second
        set_size_second = False
        set_timer.reset()
        times.clear()
        SECOND_GESTURE_MAX_LEN = int(sum(max_dist_second_gest_setting) / len(max_dist_second_gest_setting))
        SECOND_GESTURE_MAX_LEN = int(0.9 * SECOND_GESTURE_MAX_LEN)
        max_dist_second_gest_setting.clear()
        print(SECOND_GESTURE_MAX_LEN)


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
        skip = False

        if 0 <= area <= 1700:
            fingers = detector.fingers_up()
            if fingers[0]:
                if set_size_first:
                    setting_timer.start()
                    curr_finger_len = detector.find_distance(img, 6, 7)[0]
                    distance = detector.find_distance(img, 4, 8)[0]
                    set_size_first_gesture(curr_finger_len, distance, box_size, setting_timer)
                    if len(times) > 0:
                        cv2.putText(img, str(min(times)), (0, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    detector.black_color, 6)
                        skip = True
                    else:
                        skip = False
                if not skip:
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
                if set_size_second:
                    setting_timer.start()
                    dist = detector.find_distance(img, 8, 12)[0]
                    set_size_second_gesture(dist, setting_timer)
                    if len(times) > 0:
                        cv2.putText(img, str(min(times)), (0, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    detector.black_color, 6)
                        skip = True
                    else:
                        skip = False
                if not skip:
                    curr_finger_len = detector.find_distance(img, 6, 7)[0]
                    distance, circle = detector.find_distance(img, 8, 12, STANDARD_FINGER_LEN, curr_finger_len, draw=True)
                    cv2.circle(img, circle, 10, detector.white_color, cv2.FILLED)

                    if distance <= 60:
                        gesture_timer.start()
                        cv2.circle(img, circle, 25, detector.green_color, cv2.FILLED)

                    if distance > SECOND_GESTURE_MAX_LEN:
                        gesture_timer.stop()
                        curr_time = time.time()
                        if try_prev_time is not None and curr_time - try_prev_time > PREVIOUS_TRACK_TIME:
                            cv2.circle(img, circle, 10, detector.green_color, cv2.FILLED)
                            print("Next Track")
                            keyboard.send("next track")
                            timer = 0
                            text_type = 1
                            try_prev = False
                            try_prev_time = None
                            gesture_timer.reset()
                        if gesture_timer.get_time() >= PAUSE_PLAY_TIME:
                            cv2.circle(img, circle, 10, detector.green_color, cv2.FILLED)
                            print("Play/Pause")
                            keyboard.send("play/pause")
                            timer = 0
                            text_type = 3
                            try_prev = False
                            try_prev_time = None
                            gesture_timer.reset()
                        elif gesture_timer.get_time() > NEXT_TRACK_TIME:
                            if not try_prev:
                                try_prev = True
                                try_prev_time = time.time()
                                gesture_timer.reset()
                            else:
                                cv2.circle(img, circle, 10, detector.green_color, cv2.FILLED)
                                print("Previous Track")
                                keyboard.send("previous track")
                                timer = 0
                                text_type = 2
                                try_prev = False
                                try_prev_time = None
                        gesture_timer.reset()

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

    if action == ord('1'):
        if not set_size_second:
            set_size_first = True

    if action == ord('2'):
        if not set_size_first:
            set_size_second = True
