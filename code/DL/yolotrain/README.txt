Hay que seguir la estructura de directorios y el formato de etiquetado,
que en este ejemplo hacemos automáticamente con facemesh.py.

train
    images
    labels

Luego copiamos una o dos a

val
    images
    labels


yolo detect train data=boca.yaml model=yolo11n.pt epochs=200 imgsz=640 augment=True

Aunque haya pocos ejemplos, hay que hacer un número de epochs suficientes para que
el map50-95 y otras métricas suban lo suficiente.

Deja el modelo en 

/home/brutus/.pyenv/runs/detect/trainX/weights/best.pt

lo copiamos a boca.pt y lo probamos con yolo_run.py.

