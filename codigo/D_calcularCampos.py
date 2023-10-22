import numpy as np
import matplotlib.pyplot as plt
from openpiv import tools, pyprocess, validation, filters, scaling
import os
from tqdm import tqdm
import pandas as pd

def remove_velocity_outliers(u, v):
    """
    Cambiar por NaN las velocidades alejadas en módulo de la media.
    
    u, v : arrays 2D con las componentes del campo de velocidades
    """
    # Velocidad imaginaria
    velocity = u + 1j*v
    
    # Media y std de la velocidad (ignora las que no están definidas)
    mean_velocity = np.nanmean(velocity)
    sigma = np.nanstd(velocity)
    
    # Mask para velocidades alejadas
    filtered_velocity = np.where(np.abs(velocity - mean_velocity) > 3*sigma)
    
    # Cambiar velocidades alejadas por NaN
    u[filtered_velocity], v[filtered_velocity] = np.nan, np.nan
    
    return u, v

def get_velocity_field(start, stop, path, fps, pixel2cm, step=1, winsize=32, searchsize=32, overlap=16, threshold = 1.3):
    """
    start, stop : int
        Las velocidades se van a calcular desde el frame `start` hasta `stop`
    path : str
        dirección a la carpeta donde están los frames.
    fps : float
        cuadros por segundo del video
    pixel2cm : float
        factor de conversión para ir de pixeles a cm.
    winsize, searchsize, overlap : int.
        winsize = tamaños de las ventanas donde se van a calcular los desplazamientos
        overlap = superposicion entre las ventanas
        seachsize = 
    threshold : float
        cuanto más alto, mayor el filtro de los datos.

    DEVUELVE: 
    T, x, y, U, V : ndarrays
        T : array de tiempos relativos desde el frame `start`
        x, y : coordenadas de los frames.
        U, V : componentes de las velocidades en x y en y respectivamente _en el tiempo_. Dimensiones: (Nx, Ny, Nt)
    """
    dt     = 1 / fps
    frames = os.listdir(path)

    test_frame = tools.imread(path + os.sep + frames[0])  
    x, y   = pyprocess.get_coordinates(image_size = test_frame.shape, search_area_size = searchsize, overlap = overlap)
    Nx, Ny = x.shape
    Nt     = stop - start

    frame_ids  = [int(os.path.splitext(f)[0]) for f in frames]
    frame_dict = dict(zip(frame_ids, frames))

    T = np.zeros(Nt)
    U, V = np.zeros((Nt, Nx, Ny)), np.zeros((Nt, Nx, Ny))
    for n in tqdm(range(0, Nt, step)):
        frame1_id = start + n
        frame2_id = frame1_id + 1

        try: 
            frame1_file = frame_dict[frame1_id]
            frame2_file = frame_dict[frame2_id]

            frame1 = tools.imread(path + os.sep + frame1_file)
            frame2 = tools.imread(path + os.sep + frame2_file)

        except:
            raise(Exception("frame index out of range; Check: \n that the stop argument is lower than the total number of frames or \
                            \n that the name format for the image is the right one."))

        x, y = pyprocess.get_coordinates(image_size = frame1.shape, search_area_size = searchsize, overlap = overlap)
        u, v, sn = pyprocess.extended_search_area_piv(frame1.astype(np.int32), frame2.astype(np.int32), 
                                                      window_size = winsize, overlap = overlap, dt = dt, search_area_size = searchsize, 
                                                      sig2noise_method = 'peak2peak')
        
        u[sn < threshold], v[sn < threshold] = np.nan, np.nan

        x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = pixel2cm)
        x, y, u, v = tools.transform_coordinates(x, y, u, v)

        u, v = remove_velocity_outliers(u, v)

        T[n] = n * dt
        U[n] = u
        V[n] = v

    return T, x, y, U, V

def flatten_data(T, x, y, U, V):
    """
    Transforma los datos de `get_velocity_field` para obtenerlos en una dimension espacial y una temporal, en vez de dos espaciales y una temporal. Los arrays van de tener la forma (Nx, Ny, Nt) a (Nx*Ny, Nt). 
    """
    N = np.prod(np.shape(x))
    x  = np.reshape(x, N)
    y  = np.reshape(y, N)
    U  = np.array([np.reshape(u, N) for u in U])
    V  = np.array([np.reshape(v, N) for v in V])
    return T, x, y, U, V

def save_fields(dir, T, x, y, U, V, flatten = True):
    """
    Guarda los datos de `get_velocity_field` en un archivo 'campos.npz'.
    dir : str
        carpeta donde guardar 'campos.npz'
    T, x, y, U, V : ndarrays
        información del campo que sale de `get_velocity_field`
    flatten : bool
        si aplicar la funcion `flatten_data` a los datos antes de guardarlos.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

    if flatten:
        T, x, y, U, V = flatten_data(T, x, y, U, V)

    # Guardar campos como archivo comprimido de numpy
    np.savez_compressed(dir + os.sep + 'campos', T = T, x = x, y = y, U = U, V = V)

    print(f"Datos guardados en: '{dir + os.sep + 'campos.npz'}")
