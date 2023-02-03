import cv2 as cv
import os

import imutils

modelo = 'FotosAuron'
ruta1 = 'C:/Users/pms_l/Documents/Cursos/Reconocimiento Facial - Python/Codes/reconocimientofacial1'
rutaCompleta = ruta1 + '/' + modelo
if not os.path.exists(rutaCompleta):
    os.mkdir(rutaCompleta)

# camara = cv.VideoCapture(0)
camara = cv.VideoCapture("videoauron.mp4")

ruidos = cv.CascadeClassifier(
    'C:\\Users\pms_l\Documents\Cursos\Reconocimiento Facial - Python\Material\opencv\data\haarcascades\haarcascade_frontalface_default.xml')

id = 0

while True:
    respuesta, captura = camara.read()
    if not respuesta: break
    captura = imutils.resize(captura, width=640)
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idCaptura = captura.copy()

    cara = ruidos.detectMultiScale(grises, 1.3, 4)
    for (x, y, e1, e2) in cara:
        cv.rectangle(captura, (x, y), (x + e1, y + e2), (0, 255, 0), 2)
        rostroCapturado = idCaptura[y:y + e2, x:x + e1]
        rostroCapturado = cv.resize(rostroCapturado, (160, 160), interpolation=cv.INTER_CUBIC)
        cv.imwrite(rutaCompleta + '/rostro_{}.png'.format(id), rostroCapturado)
        id = id + 1

    cv.imshow("Resultado rostro", captura)

    # if cv.waitKey(1) == ord("s"):
    #     break

    if id == 350:
        break
camara.release()
cv.destroyAllWindows()
