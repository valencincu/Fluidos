#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 08:37:22 2023

@author: plopezmaggi
"""

#%%
from __future__ import print_function
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% Partir en frames
def video2images(video_path, images_path):
    """
    video_path : ruta al video
    images_path : ruta para guardar las imágenes
    """
    # Abrir dirección del video
    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(video_path))
    
    # Leer cada cuadro del video y guardarlo en la carpeta images_path
    i = 0

    if not capture.isOpened():
        raise(Exception("Unable to open file"))
    while True:
        _, frame = capture.read() ### Reads each frame of the video
        if frame is None:
            break
        max_digits = 4
        i_digits = len(str(i))
        imagen_numero = "0" * (max_digits - i_digits) + str(i) + ".jpg"

        cv2.imwrite(images_path+imagen_numero,frame)
        i += 1

#%% Esto hace lo mismo (partir en frames) pero además procesa las imágenes para mejorar el contraste
def pre_process(dir: str, file_name: str, frames_dir: str, masks_dir: str, start_at: int, stop_at: int, method = "KNN", filter_color = False, crop = None):
    """ 
    Process each individual frame of a video, generating a high contrast set of frames.
    It removes the background and light reflections on the surface of the fluid, with an optional
    feature of removing the color.
    It's configured to filter all non-blue color by default. To remove another, modify the lower_color and upper_color variables.
    
    Attributes
    ----------
    dir, file_name : str
        directory and file-name for the video you'd like to process. 
    frames_dir, masks_dir : str
        directories where the processed frames and masks will be saved.
    start_at, stop_at : int
        frame numbers at which to start and stop.
    method : str
        method used for the background subtraction. Either MOG2 of KNN. Default: KNN
    filter_color : bool
        whether to apply a color filter on top of the background subtraction. Default: False. 
        NOTE: In order for the color filter to work, you have to adjust the color parameters (lower_color, upper_color).
    crop : function
        selects the portion of the image to process if not None.

    """
    # Carpetas para guardar los cuadros (OJO: si ya existen se borran los contenidos anteriores)
    if os.path.exists(frames_dir):
        for f in os.listdir(frames_dir):
            os.remove(frames_dir + os.sep + f)
    else:
        os.makedirs(frames_dir)

    if os.path.exists(masks_dir):
        for f in os.listdir(masks_dir):
            os.remove(masks_dir + os.sep + f)
    else:
        os.makedirs(masks_dir)

    # Abre el video
    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(dir + os.sep + file_name))
    if not capture.isOpened():
        raise(Exception("Unable to open file"))

    # Determina el método para eliminar el fondo. Hay 2 métodos, MOG y KNN. KNN es un poco más rápido computacionalmente pero si el fondo es poco estable es preferible elegir MOG2. Por default si no ponemos nada va KNN. 
    if method == "MOG2":
        backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    elif method == "KNN":
        backSub = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    else:
        raise(Exception("Invalid method name"))

    for i in range(0, start_at):
        capture.read() 

    for i in tqdm(range(start_at, stop_at + 1)): # Para procesar los fotogramas uno por uno
        _, frame = capture.read() 
        if (frame is None):
            break


        fgMask = backSub.apply(frame)   # mascara para sacar el fondo
        res1 = cv2.bitwise_and(frame, frame, mask=fgMask) # saca el fondo

        res2 = res1
        if crop is not None:
            res2 = crop(res1)

        # Pasar a escala de grises, extraer el resplandor
        gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]
        res3 = cv2.inpaint(res2, mask, 21, cv2.INPAINT_TELEA)

        res4 = res3
        if filter_color:
            hsv = cv2.cvtColor(res3, cv2.COLOR_BGR2HSV)
            lower_color = np.array([0, 0, 0])
            upper_color = np.array([180, 255, 255])
            filter_mask = cv2.inRange(hsv, lower_color, upper_color)
            res4 = cv2.bitwise_and(res3, res3, mask = filter_mask)

        # # Numerar a la imagen por su numero de frame
        # cv2.rectangle(res4, (0, 0), (100,20), (255,255,255), -1)
        # cv2.putText(res4, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
        # Genera el nombre del archivo para la imagen procesada
        max_digits = max(len(str(stop_at)), 4)
        i_digits = len(str(i))
        imagen_numero = "0" * (max_digits - i_digits) + str(i) + ".jpg"

        # Guarda la mascara
        cv2.imwrite(masks_dir + os.sep + imagen_numero, mask)

        # Guarda el cuadro
        cv2.imwrite(frames_dir + os.sep + imagen_numero, res4)

    # Close all windows
    capture.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    print(f"Se guardaron los cuadros en {frames_dir}")
    print(f"Se guardaron los cuadros en {masks_dir}")
    return None

#%% Recortes posibles
def crear_recorte_circular(center, radius):
    """
    center: tuple
        tupla con las coordenadas del centro (centro_x, centro_y)
    radius: float
        valor del radio.
    """

    def recorte(frame):
        center_x, center_y = center
        x1 = center_x - radius
        y1 = center_y - radius
        x2 = center_x + radius
        y2 = center_y + radius
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        mask = np.zeros_like(frame)
        cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), thickness=-1)
        cropped_image = cv2.bitwise_and(frame, mask)
        cropped_image = cropped_image[y1:y2, x1:x2]
        return cropped_image
    
    return recorte

def crear_recorte_rectangular(esquina, horizontal, vertical):
    """
    esquina: tuple
        tupla con las coordenadas de una esquina del rectangulo.
    horizontal, vertical: float
        ancho y alto del recorte.
    """
    def recorte(frame):
        mask = np.zeros_like(frame)

        x, y = esquina
        x1, x2 = x, x + horizontal
        y1, y2 = y, y + vertical
        
        cv2.rectangle(mask, esquina, (x2, y2), (255, 255, 255), thickness = -1)
        cropped_image = cv2.bitwise_and(frame, mask)
        cropped_image = cropped_image[y1:y2, x1:x2]
        return cropped_image

    return recorte