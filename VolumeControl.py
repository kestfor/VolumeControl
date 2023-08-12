import os
import sys
import time
import json

import cv2
from cv2 import FONT_HERSHEY_SIMPLEX as FONT
import keyboard
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import HandTrackingModule as htm


class Timer:
    def __init__(self):
        self._start_time = None
        self._end_time = None
        self._started = False
        self._ended = True

    def start(self):
        if self._ended:
            self._start_time = time.time()
            self._started = True
            self._ended = False

    def stop(self):
        if self._started:
            self._end_time = time.time()
            self._ended = True
            self._started = False

    def reset(self):
        self.__init__()

    def get_time(self):
        if self._start_time is not None:
            if self._end_time is None:
                return time.time() - self._start_time
            else:
                return self._end_time - self._start_time
        else:
            return 0


WIDTH = 640
HEIGHT = 480

if len(sys.argv) == 5:
    first_name, second_name = sys.argv[1], sys.argv[3]
    if first_name == '-w' and second_name == '-h':
        WIDTH = int(sys.argv[2])
        HEIGHT = int(sys.argv[4])
elif len(sys.argv) != 1:
    raise SyntaxError("необязательные аргументы: -w, -h")

CONFIG_FILE_NAME = "config.json"
SETTING_TIME = 3
PAUSE_PLAY_TIME = 0.7
PREVIOUS_TRACK_TIME = 0.7
NEXT_TRACK_TIME = 0.1
STANDARD_BOX_HEIGHT = 329
STANDARD_BOX_WIDTH = 346
STANDARD_BOX_SIZE = [STANDARD_BOX_WIDTH, STANDARD_BOX_HEIGHT]
FIRST_GESTURE_MAX_LEN = 220
FIRST_GESTURE_MIN_LEN = 25
SECOND_GESTURE_MAX_LEN = 79
STANDARD_REFERENCE_FINGER_LEN = 73


setting_mode = False
set_size_first = False
set_size_second = False

reference_finger_len_for_setting = []
max_dist_for_setting = []
box_size_for_setting = []
max_dist_second_gest_setting = []
times = set()

error_range = 0.4
config_dict = {}


def get_config(reference_ratio: float) -> int:
    global config_dict
    if config_dict:
        global STANDARD_BOX_SIZE
        global FIRST_GESTURE_MAX_LEN
        global STANDARD_REFERENCE_FINGER_LEN
        global SECOND_GESTURE_MAX_LEN
        min_delta = 100
        user_id = None
        for key in config_dict:
            if reference_ratio - config_dict[key]["REFERENCE_RATIO"] < min_delta:
                min_delta = reference_ratio - config_dict[key]["REFERENCE_RATIO"]
                user_id = key
        STANDARD_BOX_SIZE = config_dict[user_id]["STANDARD_BOX_SIZE"]
        FIRST_GESTURE_MAX_LEN = config_dict[user_id]["FIRST_GESTURE_MAX_LEN"]
        STANDARD_REFERENCE_FINGER_LEN = config_dict[user_id]["STANDARD_REFERENCE_FINGER_LEN"]
        SECOND_GESTURE_MAX_LEN = config_dict[user_id]["SECOND_GESTURE_MAX_LEN"]
        return user_id


def load_config(file_name: str) -> None:
    if os.path.exists(file_name):
        with open(file_name, "r") as file:
            global config_dict
            tmp_dict = dict(json.load(file))
            for key, value in tmp_dict.items():
                config_dict[int(key)] = value


def update_config(file_name: str) -> None:
    new_key = -1
    min_error = 100
    for key in config_dict:
        new_key = max(key, new_key)
        min_error = min(min_error, config_dict[key]["REFERENCE_RATIO"])
    new_key += 1
    if (not config_dict) or min_error < error_range:
        new_settings = {"STANDARD_BOX_SIZE": STANDARD_BOX_SIZE,
                        "FIRST_GESTURE_MAX_LEN": FIRST_GESTURE_MAX_LEN,
                        "STANDARD_REFERENCE_FINGER_LEN": STANDARD_REFERENCE_FINGER_LEN,
                        "SECOND_GESTURE_MAX_LEN": SECOND_GESTURE_MAX_LEN,
                        "REFERENCE_RATIO": FIRST_GESTURE_MAX_LEN / STANDARD_REFERENCE_FINGER_LEN}
        config_dict[new_key] = new_settings
        with open(file_name, "w") as file:
            json.dump(config_dict, file, sort_keys=True, indent=4)
        print("added new user")


def average(items: [list, tuple]) -> float:
    return sum(items) / len(items)


def set_size_first_gesture(img, detector: htm.HandDetector, hand_box: [list, tuple], set_timer: Timer) -> None:
    global times
    curr_time = int(set_timer.get_time())
    curr_dist = detector.find_distance(img, 4, 8)[0]
    reference_finger_len = detector.find_distance(img, 1, 2)[0]
    reference_finger_len_for_setting.append(reference_finger_len)
    max_dist_for_setting.append(curr_dist)
    box_size_for_setting.append(hand_box)

    times.add(SETTING_TIME - curr_time)

    if curr_time >= SETTING_TIME:
        global STANDARD_REFERENCE_FINGER_LEN
        global FIRST_GESTURE_MAX_LEN
        global STANDARD_BOX_SIZE
        global set_size_first
        global set_size_second
        set_size_first = False
        set_size_second = True
        set_timer.reset()
        times.clear()
        STANDARD_REFERENCE_FINGER_LEN = int(average(reference_finger_len_for_setting))
        FIRST_GESTURE_MAX_LEN = int(average(max_dist_for_setting))
        STANDARD_BOX_SIZE[0] = int(average([item[0] for item in box_size_for_setting]))
        STANDARD_BOX_SIZE[1] = int(average([item[1] for item in box_size_for_setting]))
        reference_finger_len_for_setting.clear()
        max_dist_for_setting.clear()
        box_size_for_setting.clear()


def set_size_second_gesture(dist, set_timer: Timer) -> None:
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
        SECOND_GESTURE_MAX_LEN = int(average(max_dist_second_gest_setting))
        SECOND_GESTURE_MAX_LEN = int(0.9 * SECOND_GESTURE_MAX_LEN)
        max_dist_second_gest_setting.clear()
        update_config(CONFIG_FILE_NAME)


def get_text(type_of_text: int) -> str:
    if type_of_text == 1:
        return "Next Track"
    if type_of_text == 2:
        return "Previous Track"
    if type_of_text == 3:
        return "Play/Pause"


def start():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
    min_volume, max_volume, a = volume.GetVolumeRange()

    global WIDTH, HEIGHT
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    detector = htm.HandDetector(max_num_hands=1, min_detection_confidence=0.9, min_tracking_confidence=0.9)
    curr_volume = np.interp(volume.GetMasterVolumeLevel(), (min_volume, max_volume), (0, 100))
    try_prev = False
    try_prev_time = None
    enable_fps = False
    enable_draw = False
    text_type = 0
    timer = 0
    prev_time = 0

    hand_in_frame = False

    gesture_timer = Timer()
    setting_timer = Timer()
    global set_size_first, set_size_second, times, setting_mode

    while True:

        success, img = cap.read()
        img = cv2.flip(img, 1)

        detector.set_hands(img, draw=enable_draw)
        landmarks, hand_box = detector.find_position(img, draw=enable_draw)

        if len(landmarks) > 0:
            if not hand_in_frame:
                curr_dist = detector.find_distance(img, 4, 8)[0]
                finger_len = detector.find_distance(img, 1, 2)[0]
                reference_ratio = curr_dist / finger_len
                user_id = get_config(reference_ratio)
                if user_id is not None and not setting_mode:
                    print(f'user_id: {user_id}')
                hand_in_frame = True
        else:
            hand_in_frame = False

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        if enable_fps:
            cv2.putText(img, str(int(fps)), (0, 30), FONT, 1, detector.black_color, 3)

        if hand_in_frame:
            box_width = abs(hand_box[0] - hand_box[2])
            box_height = abs(hand_box[1] - hand_box[3])
            area = box_height * box_width // 100
            box_size = (box_width, box_height)

            if 0 <= area <= 1700:
                fingers = detector.fingers_up()
                if fingers[0]:
                    if set_size_first:
                        setting_timer.start()
                        set_size_first_gesture(img, detector, box_size, setting_timer)
                        if len(times) > 0:
                            cv2.putText(img, str(min(times)), (0, HEIGHT - 15), FONT, 1, detector.black_color, 6)
                    if not setting_mode:
                        curr_finger_len = detector.find_distance(img, 1, 2)[0]
                        distance, circle = detector.find_distance(img, 4, 8, STANDARD_REFERENCE_FINGER_LEN,
                                                                  curr_finger_len, draw=True)
                        cv2.circle(img, circle, 10, detector.white_color, cv2.FILLED)
                        volume_percentage = np.interp(distance,
                                                      (FIRST_GESTURE_MIN_LEN, FIRST_GESTURE_MAX_LEN), (0, 100))
                        smoothness = 3
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
                            cv2.putText(img, str(min(times)), (0, HEIGHT - 15), FONT, 1,
                                        detector.black_color, 6)
                    if not setting_mode:
                        curr_finger_len = detector.find_distance(img, 1, 2)[0]
                        distance, circle = detector.find_distance(img, 8, 12, STANDARD_REFERENCE_FINGER_LEN,
                                                                  curr_finger_len, draw=True)
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
                            if gesture_timer.get_time() >= PAUSE_PLAY_TIME:
                                cv2.circle(img, circle, 10, detector.green_color, cv2.FILLED)
                                print("Play/Pause")
                                keyboard.send("play/pause")
                                timer = 0
                                text_type = 3
                                try_prev = False
                                try_prev_time = None
                            elif gesture_timer.get_time() > NEXT_TRACK_TIME:
                                if not try_prev:
                                    try_prev = True
                                    try_prev_time = time.time()
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
            cv2.putText(img, get_text(text_type), (0, HEIGHT - 15), FONT, 1, detector.black_color, 6)
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

        if action == ord('s'):
            if setting_mode:
                set_size_first = False
            else:
                set_size_first = True
            setting_mode = not setting_mode


def main():
    load_config(CONFIG_FILE_NAME)
    start()


if __name__ == "__main__":
    main()
