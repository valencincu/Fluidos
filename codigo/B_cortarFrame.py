#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Wed Sep  6 17:50:22 2023

@author: plopezmaggi

Corta un frame de un video. Asume que el video está en la carpeta que Spyder tiene abierta, y guarda ahí la imagen.

"""
#%%
import os
import cv2
import matplotlib.pyplot as plt

def get_frame_and_properties(video_path, save_as, N):
    """
    video_path : str
        ruta al video.
    save_as : str
        nombre del frame guardado.
    N : int
        frame a recortar.
    DEVUELVE:
    frame : ndarray
        Array con la información del frame, tiene dimensiones (Ny, Nx, 3).
    properties : dict
        diccionario con propiedades del video, en particular 
            - 'fps': cuadros por segundo
            - 'dimensions': dimensiones de los cuadros
            - 'num_frames': numero de frames que tiene el video.
    """
    video = cv2.VideoCapture(video_path)
    num_frames = video.get((cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get((cv2.CAP_PROP_FPS))
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # 3
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame = video.read()[1]
    for i in range(N):  # para devolver el frame numero N
        frame = video.read()[1]
    cv2.imwrite(save_as, frame)
    print(f"Se guardo el cuadro en {save_as}.")
    properties = {
        "fps": fps,
        "num_frames": num_frames,
        "dimensions": (width, height)
    }
    return frame, properties
