import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import json
import random
import pickle
import spicy

with open("contenido.json", encoding='utf-8') as archivo:
    datos = json.load(archivo)
try:
    with open("variables.pickle","rb") as archivoPickle:
        palabras, tags, entrenamiento, salida = pickle.load(archivoPickle)
except:
    palabras=[]
    tags=[]
    auxX=[]
    auxY=[]

    for contenido in datos["contenido"]:
        for patrones in contenido["patrones"]:
            auxPal = nltk.word_tokenize(patrones) #separa en tokens las palabras de los mensajes
            palabras.extend(auxPal)
            auxX.append(auxPal)
            auxY.append(contenido["tag"])

            if contenido["tag"] not in tags:
                tags.append(contenido["tag"])

    palabras = [stemmer.stem(w.lower()) for w in palabras if w!="?"]
    palabras = sorted(list(set(palabras)))
    tags = sorted(tags)

    entrenamiento = []
    salida=[]
    salidaVacia=[0 for _ in range(len(tags))]

    for x, documento in enumerate(auxX):
        cubeta = []
        auxPalabra = [stemmer.stem(w.lower())for w in documento]
        for w in palabras:
            if w in auxPalabra:
                cubeta.append(1)
            else:
                cubeta.append(0)
        filaSalida = salidaVacia[:]
        filaSalida[tags.index(auxY[x])]=1
        entrenamiento.append(cubeta)
        salida.append(filaSalida)
    #red neuronal

    entrenamiento = numpy.array(entrenamiento)
    salida = numpy.array(salida)
    with open("variables.pickle","wb") as archivoPickle:
        pickle.dump((palabras, tags, entrenamiento, salida))
from tensorflow.python.framework import ops
ops.reset_default_graph() # 

red = tflearn.input_data(shape=[None,len(entrenamiento[0])])
red = tflearn.fully_connected(red,10)
red = tflearn.fully_connected(red,10)
red = tflearn.fully_connected(red,len(salida[0]),activation="softmax")
red = tflearn.regression(red)

modelo = tflearn.DNN(red)
modelo.fit(entrenamiento,salida,n_epoch=1000,batch_size=10,show_metric=True)#entre mas veces lo vea mejor
modelo.save("modelo.tflearn")

def bot():
    while True:
        entrada = input("Tu: ")
        cubeta = [0 for _ in range(len(palabras))]
        entradaProcesada = nltk.word_tokenize(entrada)
        entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]
        for palabraInd in entradaProcesada:
            for i,palabra in enumerate(palabras):
                if(palabra == palabraInd):
                    cubeta[i] = 1
        resultados = modelo.predict([numpy.array(cubeta)])
        resultadosIndices = numpy.argmax(resultados)
        tag = tags[resultadosIndices]

        for tagAux in datos["contenido"]:
            if tagAux["tag"] == tag:
                respuesta = tagAux["respuestas"]

        print("BOT: ",random.choice(respuesta))
bot()