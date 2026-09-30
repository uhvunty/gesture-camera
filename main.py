import cv2
import mediapipe as mp
import urllib.request
import os


# ===========================================================
# 1. DOWNLOAD THE MEDIAPIPE HAND MODEL
# ===========================================================

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe hand model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded!")


# ============================================================
# 2. SET UP MEDIAPIPE
# ============================================================

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)


# ============================================================
# 3. HAND CONNECTIONS
# ============================================================

# MediaPipe hand landmark connections

connections = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (5, 9),
    (9, 13),
    (13, 17)
]


# ============================================================
# 4. FINGER NAMES
# ============================================================

finger_names = [
    "Thumb",
    "Index",
    "Middle",
    "Ring",
    "Pinky"
]


# ============================================================
# 5. FUNCTION TO CHECK WHICH FINGERS ARE UP
# ============================================================

def get_fingers_up(points, handedness):

    fingers = []

    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    if points[8][1] < points[6][1]:
        fingers.append("Index")

    # --------------------------------------------------------
    # MIDDLE
    # --------------------------------------------------------

    if points[12][1] < points[10][1]:
        fingers.append("Middle")

    # --------------------------------------------------------
    # RING
    # --------------------------------------------------------

    if points[16][1] < points[14][1]:
        fingers.append("Ring")

    # --------------------------------------------------------
    # PINKY
    # --------------------------------------------------------

    if points[20][1] < points[18][1]:
        fingers.append("Pinky")

    # --------------------------------------------------------
    # THUMB
    # --------------------------------------------------------

    # MediaPipe gives Left/Right information.
    # Since the webcam image is mirrored, this works
    # reasonably for a normal front-facing hand.

    if handedness == "Right":

        if points[4][0] < points[3][0]:
            fingers.append("Thumb")

    else:

        if points[4][0] > points[3][0]:
            fingers.append("Thumb")

    return fingers


# ============================================================
# 6. RECOGNIZE BASIC GESTURES
# ============================================================

def recognize_gesture(fingers):

    count = len(fingers)

    # --------------------------------------------------------
    # FIST
    # --------------------------------------------------------

    if count == 0:
        return "FIST"

    # --------------------------------------------------------
    # OPEN PALM
    # --------------------------------------------------------

    if count == 5:
        return "OPEN PALM"

    # --------------------------------------------------------
    # THUMBS UP
    # --------------------------------------------------------

    if fingers == ["Thumb"]:
        return "THUMBS UP"

    # --------------------------------------------------------
    # POINTING
    # --------------------------------------------------------

    if fingers == ["Index"]:
        return "POINTING"

    # --------------------------------------------------------
    # PEACE
    # --------------------------------------------------------

    if set(fingers) == {"Index", "Middle"}:
        return "PEACE"

    # --------------------------------------------------------
    # ROCK
    # --------------------------------------------------------

    if set(fingers) == {"Index", "Pinky"}:
        return "ROCK"

    # --------------------------------------------------------
    # THREE
    # --------------------------------------------------------

    if count == 3:
        return "THREE FINGERS"

    # --------------------------------------------------------
    # FOUR
    # --------------------------------------------------------

    if count == 4:
        return "FOUR FINGERS"

    # --------------------------------------------------------
    # OTHERWISE
    # --------------------------------------------------------

    return "UNKNOWN"


# ============================================================
# 7. OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    detector.close()

    exit()


# ============================================================
# 8. MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read webcam.")
        break


    # --------------------------------------------------------
    # MIRROR CAMERA
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)


    # --------------------------------------------------------
    # CONVERT BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # CREATE MEDIAPIPE IMAGE
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # DETECT HANDS
    # --------------------------------------------------------

    result = detector.detect(mp_image)


    # ========================================================
    # PROCESS EVERY DETECTED HAND
    # ========================================================

    if result.hand_landmarks:

        for hand_number, hand_landmarks in enumerate(
            result.hand_landmarks
        ):

            # ------------------------------------------------
            # GET IMAGE SIZE
            # ------------------------------------------------

            height, width, _ = frame.shape


            # ------------------------------------------------
            # CONVERT NORMALIZED COORDINATES TO PIXELS
            # ------------------------------------------------

            points = []

            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))


            # ------------------------------------------------
            # GET LEFT / RIGHT HAND
            # ------------------------------------------------

            handedness = "Unknown"

            if result.handedness:

                handedness = result.handedness[
                    hand_number
                ][0].category_name


            # =================================================
            # DRAW CONNECTION LINES
            # =================================================

            for start, end in connections:

                x1, y1 = points[start]
                x2, y2 = points[end]

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )


            # =================================================
            # DRAW ALL 21 LANDMARKS
            # =================================================

            for i, (x, y) in enumerate(points):

                cv2.circle(
                    frame,
                    (x, y),
                    6,
                    (0, 0, 255),
                    -1
                )

                # Landmark number

                cv2.putText(
                    frame,
                    str(i),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1
                )


            # =================================================
            # CHECK FINGERS
            # =================================================

            fingers_up = get_fingers_up(
                points,
                handedness
            )


            # =================================================
            # RECOGNIZE GESTURE
            # =================================================

            gesture = recognize_gesture(
                fingers_up
            )


            # =================================================
            # FINGER LABELS
            # =================================================

            finger_tips = {
                "Thumb": 4,
                "Index": 8,
                "Middle": 12,
                "Ring": 16,
                "Pinky": 20
            }


            for finger, tip_index in finger_tips.items():

                x, y = points[tip_index]

                cv2.putText(
                    frame,
                    finger,
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2
                )


            # =================================================
            # HAND INFORMATION
            # =================================================

            wrist_x, wrist_y = points[0]

            cv2.putText(
                frame,
                handedness + " Hand",
                (wrist_x - 30, wrist_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


            # =================================================
            # GESTURE TEXT
            # =================================================

            cv2.putText(
                frame,
                "Gesture: " + gesture,
                (20, 45 + hand_number * 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                3
            )


            # =================================================
            # FINGER COUNT
            # =================================================

            cv2.putText(
                frame,
                "Fingers: " + str(len(fingers_up)),
                (20, 80 + hand_number * 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


    # ========================================================
    # DISPLAY CAMERA
    # ========================================================

    cv2.imshow(
        "Gesture Camera",
        frame
    )


    # ========================================================
    # PRESS Q TO QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 9. CLEAN UP
# ============================================================

cap.release()

cv2.destroyAllWindows()

detector.close()

