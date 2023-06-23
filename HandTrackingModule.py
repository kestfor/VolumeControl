try:
    import mediapipe as mp
    import cv2
    import math
    import numpy
    from google.protobuf.json_format import MessageToDict
except ImportError:
    print('requires "opencv", "mediapipe" packages.')
    print('Install it via command:')
    print('    pip install numpy')
    raise


class HandDetector:
    __BLACK = (0, 0, 0)
    __WHITE = (255, 255, 255)
    __RED = (0, 0, 255)
    __BLUE = (255, 0, 0)
    __GREEN = (0, 255, 0)
    __CUSTOM = (0, 0, 0)

    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.__mode = static_image_mode
        self.__max_hands = max_num_hands
        self.__model_complexity = model_complexity
        self.__detection_confidence = min_detection_confidence
        self.__tracking_confidence = min_tracking_confidence
        self.__mpHands = mp.solutions.hands
        self.__hands = self.__mpHands.Hands(self.__mode, self.__max_hands, self.__model_complexity,
                                            self.__detection_confidence, self.__tracking_confidence)
        self.__mpDraw = mp.solutions.drawing_utils
        self.__results = None
        self.__is_right = False
        self.__is_left = False
        self.__tip_ids = (4, 8, 12, 16, 20)
        self.__landmarks_list = None

    @property
    def black_color(self):
        return self.__BLACK

    @property
    def custom_color(self):
        return self.__CUSTOM

    @custom_color.setter
    def custom_color(self, new_color: tuple):
        if type(new_color) not in (tuple, list):
            raise TypeError
        else:
            self.__CUSTOM = new_color

    @property
    def white_color(self):
        return self.__WHITE

    @property
    def green_color(self):
        return self.__GREEN

    @property
    def red_color(self):
        return self.__RED

    @property
    def blue_color(self):
        return self.__BLUE

    def __get_vect(self, first_landmark, second_landmark):
        return self.__landmarks_list[second_landmark][1] - self.__landmarks_list[first_landmark][1],\
               self.__landmarks_list[second_landmark][2] - self.__landmarks_list[first_landmark][2]

    def set_hands(self, image, draw=True):

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.__results = self.__hands.process(img_rgb)
        if self.__results.multi_hand_landmarks:
            for i in self.__results.multi_handedness:
                label = MessageToDict(i)['classification'][0]['label']
                if label == "Right":
                    self.__is_right = True
                    self.__is_left = False
                else:
                    self.__is_right = False
                    self.__is_left = True
        if draw:
            if self.__results.multi_hand_landmarks is not None:
                for handLms in self.__results.multi_hand_landmarks:
                    self.__mpDraw.draw_landmarks(image, handLms, self.__mpHands.HAND_CONNECTIONS)

        return image

    def find_position(self, image, hand_number=0, draw=True):
        x_list = []
        y_list = []
        self.__landmarks_list = []
        hand_box = []
        if self.__results.multi_hand_landmarks is not None:
            if len(self.__results.multi_hand_landmarks) > hand_number:
                hand = self.__results.multi_hand_landmarks[hand_number]
                for number, landmark in enumerate(hand.landmark):
                    height, width, c = image.shape
                    converted_x, converted_y = int(width*landmark.x), int(height*landmark.y)
                    self.__landmarks_list.append((number, converted_x, converted_y))
                    x_list.append(converted_x)
                    y_list.append(converted_y)
                    if draw:
                        cv2.circle(image, (converted_x, converted_y), 3, self.__GREEN, cv2.FILLED)
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            hand_box = (x_min, y_min, x_max, y_max)

            if draw:
                cv2.rectangle(image,
                              (hand_box[0] - 20, hand_box[1] - 20),
                              (hand_box[2] + 20, hand_box[3] + 20),
                              self.__GREEN,
                              3)

        return self.__landmarks_list, hand_box

    def fingers_up(self):
        fingers = []

        # thumb (new version (scalar))
        # first = numpy.array(self.__get_vect(2, 3))
        # second = numpy.array(self.__get_vect(3, 4))
        # print(numpy.dot(first, second) / (numpy.linalg.norm(first) * numpy.linalg.norm(second)))
        # if numpy.dot(first, second) / (numpy.linalg.norm(first) * numpy.linalg.norm(second)) > 0.90:
        #     fingers.append(1)
        # else:
        #     fingers.append(0)

        # thumb (old version)
        if self.__is_right:
            if self.__landmarks_list[self.__tip_ids[0]][1] < self.__landmarks_list[self.__tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if self.__landmarks_list[self.__tip_ids[0]][1] > self.__landmarks_list[self.__tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        # rest
        for num in range(1, 5):
            if self.__landmarks_list[self.__tip_ids[num]][2] < self.__landmarks_list[self.__tip_ids[num] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def find_distance(self, img, first_landmark: int, second_landmark: int,
                      standard_size=None, current_size=None, draw=False):

        if current_size is None and standard_size is None:
            current_size = 1
            standard_size = 1

        if type(standard_size) not in (tuple, list, int, float):
            raise TypeError("Unacceptable type of standard size or current size")

        if type(standard_size) not in (tuple, list):
            standard_size = (standard_size, standard_size)
            current_size = (current_size, current_size)

        if type(standard_size) != type(current_size):
            raise TypeError("Standard size and current size must have same type")

        x1, y1 = self.__landmarks_list[first_landmark][1:]
        x2, y2 = self.__landmarks_list[second_landmark][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 10, self.__WHITE, cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, self.__WHITE, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), self.__WHITE, 3)
        distance = math.hypot((x2 - x1) * (standard_size[0] / current_size[0]),
                              (y2 - y1) * (standard_size[1] / current_size[0]))
        return distance, (cx, cy)
