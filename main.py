import cv2

face_ref = cv2.CascadeClassifier("face_ref.xml")
camera = cv2.VideoCapture(0)

def facedec(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=3, minNeighbors=5, minSize=(30, 30))
    return faces

def drawerBox(frame):
    for x, y, w, h in facedec(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 4)

def close():
   camera.release()
   cv2.destroyAllWindows()
   exit()

def main():
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        drawerBox(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            close()

if __name__ == "__main__":
    main()
