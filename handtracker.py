import cv2
import mediapipe as mp
import time

# initialize camera (webcam) from ID 0 (could be a different ID if multiple webcams)
cap = cv2.VideoCapture(0)

# initialize hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

prev_time, current_time = 0, 0

# main video loop
while True:
    # get frame from camera
    success, frame = cap.read()

    # process RGB frame
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks is not None: # at least one hand detected
        # loop over all the hands
        for hand_landmarks in result.multi_hand_landmarks:

            # convert fractional landmark coordinates into pixel coordinates
            for id_, lm in enumerate(hand_landmarks.landmark):
                height, width, _ = frame.shape
                pixel_x, pixel_y = int(lm.x * width), int(lm.y * height)

            # draw the hand on the frame
            mp_draw.draw_landmarks(
                frame, hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
            )
    
    # calculate frame rate (fps)
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # display frame and other info
    cv2.putText(
        frame, f'FPS:{fps:3.0f}',
        (10, 20),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=1,
        color=(0, 0, 255),
        thickness=2,
    )
    cv2.imshow('img', frame)
    time.sleep(0.04) # optional

    # break if key 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# release camera and close all windows
cap.release()
cv2.destroyAllWindows()
