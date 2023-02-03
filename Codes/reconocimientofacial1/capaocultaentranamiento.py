import cv2 as cv
import os
import numpy as np
from time import time

dataRuta = 'C:/Users/pms_l/Documents/Cursos/Reconocimiento Facial - Python/Codes/reconocimientofacial1/Data'
listaData = os.listdir(dataRuta)

ids = []
rostrosData = []
id = 0
tiempo_inicial_lectura = time()

for fila in listaData:
    rutaCompleta = dataRuta + '/' + fila
    print("Iniciando Lectura....")
    for foto in os.listdir(rutaCompleta):
        print("Imagenes: ", fila + '/' + foto)
        ids.append(id)
        rostrosData.append(cv.imread(rutaCompleta + '/' + foto, 0))
        imagenes = cv.imread(rutaCompleta + '/' + foto, 0)
    id = id + 1

    tiempo_final_lectura = time()
    tiempo_total_lectura = tiempo_final_lectura - tiempo_inicial_lectura
    print("Tiempo Total de lectura : " ,tiempo_total_lectura)

entrenamiento_EigenFaceRecognizer = cv.face.EigenFaceRecognizer_create()
print("Iniciando el entrenamiento....espere")

tiempo_inicial_entrenamiento = time()
entrenamiento_EigenFaceRecognizer.train(rostrosData, np.array(ids))
tiempo_final_entramiento = time()

print("Tiempo de Entrenamiento Total : ", tiempo_final_entramiento - tiempo_inicial_entrenamiento)
entrenamiento_EigenFaceRecognizer.write('Entrenamiento_EigenFaceRecognizer.xml')
print("Entrenamiento concluido")
