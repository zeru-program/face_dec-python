import cv2
import mediapipe as mp

# Initialize MediaPipe hand tracking components
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Capture video from the webcam
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()

while True:
    success, image = cap.read()  # Read frame from webcam
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip and convert the image to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Process the image and detect hand landmarks
    results = hands.process(image)

    # Convert back to BGR for display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw hand landmarks if any are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

    # Display the image
    cv2.imshow('Hand Tracker by zeru', image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
