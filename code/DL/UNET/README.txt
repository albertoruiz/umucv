1) Elegimos un conjunto de imágenes que contienen las regiones de interés y las
   copiamos a la carpeta de trabajo.

2) Para cada una, ejecutamos ./masker.py <filename>.
   Marcamos con el ratón las zonas de interés.
   (se pueden borrar las últimas marcas con la tecla x).
   Al salir, se genera una nueva imagen mask_<filename> con la salida deseada
   para cada imagen.
   Las guardamos en una carpeta p.ej. 'caras'
   
3) Ejecutamos ./prepare.py --examples=<carpeta> --show, que generará
   aleatoriamente recortes de las imágenes para tener ejemplos representativos.
   Si tiene pinta de ir bien lo ejecutamos otra vez sin --show para que
   guarde los recortes en el archivo samples.npz. Lo renombramos a p.ej. caras.npz

4) Ejecutamos ./train.py --samples=caras.npz, que efectuará 20 epochs de
   optimización partiendo de cero, y guardará el modelo 'model.torch'.
   Lo renombramos a p.ej. 'caras.torch'.
   Seguramente este entrenamiento será insuficiente, por lo que volveremos a hacerlo
   varias veces, pasando el nombre del modelo que queremos refinar:
   ./train.py --samples=caras.nzp --model=caras.torch
   Al terminar cada ejecución muestra la curva de aprendizaje.
   
   De vez en cuando podemos ejecutar check.py --model=caras.torch --samples=caras.npz 
   para ver qué tal se porta en un subconjunto aleatorio de los recortes.
   
5) Cuando estemos satisfechos, podemos ejecutarlo con la cámara en vivo:
    ./rununet --model=<model>
    
    Con --dev dir:filenames*de*ejemplo comprobamos qué tal resuelve
    las imágenes completas de partida.
