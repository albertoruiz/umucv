# Es necesario instalar el siguiente paquete:

pip install rembg[cpu]

# Pero tiene el problema de que añade una versión de opencv que no nos sirve.
# Para arreglarlo ejecuta:

pip uninstall opencv-headless-python
pip uninstall opencv-contrib-python
pip install opencv-contrib-python

