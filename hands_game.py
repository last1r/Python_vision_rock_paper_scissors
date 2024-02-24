import mediapipe as mp
import cv2
from hands_variables import *

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands

def condition_rock(ip, it, mp, mt, rp, rt, pp, pt):
    # print((ip < it), (mp < mt), (rp < rt), (pp < pt))
    if (ip < it) and (mp < mt) and (rp < rt) and (pp < pt):
        return True
    return False


def condition_paper(ip, it, mp, mt, rp, rt, pp, pt):
    if (ip > it) and (mp > mt) and (rp > rt) and (pp > pt):
        return True
    return False

def condition_scissors(ip, it, mp, mt, rp, rt, pp, pt):
    if (ip > it) and (mp > mt) and (rp < rt) and (pp < pt):
        return True
    return False

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            hn = results.multi_hand_landmarks[0]
            ip = hn.landmark[INDEX_FINGER_PIP].y
            it = hn.landmark[INDEX_FINGER_TIP].y
            mp = hn.landmark[MIDDLE_FINGER_PIP].y
            mt = hn.landmark[MIDDLE_FINGER_TIP].y
            rp = hn.landmark[RING_FINGER_PIP].y
            rt = hn.landmark[RING_FINGER_TIP].y
            pp = hn.landmark[PINKY_PIP].y
            pt = hn.landmark[PINKY_TIP].y
            txt = ''
            if (condition_rock(ip, it, mp, mt, rp, rt, pp, pt)):
                r_img = cv2.imread('photo/rock.jpg')
                r_width, r_height, _ = r_img.shape
                frame[0:r_width, 0:r_height] = r_img
                txt = 'rock'
            elif (condition_paper(ip, it, mp, mt, rp, rt, pp, pt)):
                r_img = cv2.imread('photo/paper.jpg')
                r_width, r_height, _ = r_img.shape
                frame[0:r_width, 0:r_height] = r_img
                txt = 'paper'
            elif (condition_scissors(ip, it, mp, mt, rp, rt, pp, pt)):
                r_img = cv2.imread('photo/scissors.jpg')
                r_width, r_height, _ = r_img.shape
                frame[0:r_width, 0:r_height] = r_img
                txt = 'scissors'

            new_image = cv2.putText(
                img=frame,
                text=txt,
                org=(200, 200),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=3.0,
                color=(125, 246, 55),
                thickness=3
            )
        cv2.imshow('fin', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
