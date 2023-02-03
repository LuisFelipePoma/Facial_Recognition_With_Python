import cv2 as cv
import os


camara = cv.VideoCapture(0)
ruidos = cv.CascadeClassifier(
    'C:\\Users\pms_l\Documents\Cursos\Reconocimiento Facial - Python\Material\opencv\data\haarcascades\haarcascade_frontalface_default.xml')

while True:
    _, captura = camara.read()
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    cara = ruidos.detectMultiScale(grises, 1.3, 4)
    for (x, y, e1, e2) in cara:
        cv.rectangle(captura, (x, y), (x + e1, y + e2), (0, 255, 0), 2)
    cv.imshow("Resultado rostro", captura)
    if cv.waitKey(1) == ord("s"):
        break
camara.release()
cv.destroyAllWindows()
