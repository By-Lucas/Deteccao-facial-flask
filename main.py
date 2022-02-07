from flask import Flask, render_template, Response
import cv2, os
from flask import Flask 

# INSTALAR BIBLIOTECAS
# pip install flask==1.1.4
# pip install opencv-python


app = Flask(__name__)

video = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier()
face_cascade.load(cv2.samples.findFile("static/haarcascade_frontalface_alt2.xml"))

# para uso de câmera de cftv: rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# Usar WEBCAM local: cv2.VideoCapture(0)

def gerar_frames():  # gerar frame a frame da câmera
    save_path = os.path.join('users')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    imgs = []

    while True:
        success, image = video.read()
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        faces = face_cascade.detectMultiScale(frame_gray)

        for (x, y, w, h) in faces:
            center = (x + w//2, y + h//2)
            cv2.putText(image, "X: " + str(center[0]) + " Y: " + str(center[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            faceROI = frame_gray[y:y+h, x:x+w]
        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video')
def video_captura():
    # Rota de captura de vídeo. Coloque isso no atributo src de uma tag img
    return Response(gerar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

"""Página inicial de captura de vídeo."""
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)