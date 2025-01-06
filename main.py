import cv2

face_ref = cv2.CascadeClassifier("fave_ref.xml")
camera = cv2.VideoCapture(0)

def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minSize=(300, 300), minNeighbors=5)
    return faces

def drawer_box(frame):
    for x, y, w, h in face_detection(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
        
        # Tambahkan teks di bawah kotak
        name = "- Mumtaaz"  # Ganti dengan nama Anda
        age = 17  # Ganti dengan umur Anda
        text = f"{name}, {age} age"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (0, 255, 0)
        
        # Tentukan posisi teks di bawah kotak
        text_x = x
        text_y = y + h + 20  # 20 piksel di bawah kotak
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)


def exit_window():
    camera.release()
    cv2.destroyAllWindows
    exit()

def main():
    while True:
        _, frame = camera.read()
        drawer_box(frame)
        cv2.imshow("AppFace AI", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_window()

if __name__ == '__main__':
    main()