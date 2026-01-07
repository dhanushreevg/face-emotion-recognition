from facial_emotion_recognition import EmotionRecognition
import cv2

er = EmotionRecognition(device='cpu')
cam = cv2.VideoCapture(0)

frame_skip = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame_skip += 1
    if frame_skip % 3 != 0:
        cv2.imshow("Facial Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # Resize frame (IMPORTANT for speed)
    frame = cv2.resize(frame, (640, 480))

    frame = er.recognise_emotion(frame, return_type='BGR')
    cv2.imshow("Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cam.release()
cv2.destroyAllWindows()
