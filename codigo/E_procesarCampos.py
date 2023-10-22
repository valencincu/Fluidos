import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import colormaps
import os

def load_data(path):
    """
    Devuelve los datos del campo (`T`, `x`, `y`, `U`, `V`) que se encuentran en `path`. 
    """
    data = np.load(path)
    T, x, y, U, V = data["T"], data["x"], data["y"], data["U"], data["V"]
    return T, x, y, U, V

def frame_interval_average(T, x, y, U, V, frame_interval = None):
    """
    Toma los datos de los campos para todos los frames y los devuelve promediados cada `frame_interval` frames. Por ejemplo, si  originalmente `len(T) = 10` y `frame_interval = 2`, los datos que devuelve van a tener una dimensión temporal igual a 5 (promediando de dos en dos). Por default, promedia sobre todo el tiempo. 

    T, x, y, U, V : ndarrays
        Datos del campo que se obtienen de `get_velocity_field` o `load_data`.
    frame_interval : int
        Longitud de los intervalos de frames donde se van a realizar los promedios.

    DEVUELVE:
    T_avg : ndarray
        Array con el promedio de tiempos en los intervalos de longitud `frame_interval`. Si  `frame_interval = None`, devuelve solo un valor de tiempo.
    x, y : ndarays
        Arrays con las coordenadas x e y del campo.
    U_avg, V_avg : ndarrays
        Promedios de los componentes de las velocidades en los intervalos de longitud `frame_interval`.
    U_err, V_err : ndarrays
        Desviaciones estandar de los componentes de las velocidades en los intervalos de longitud `frame_interval`.
    """
    if frame_interval == None:
        frame_interval = len(T)

    U_avg, V_avg, T_avg = [], [], []
    U_err, V_err = [], []
    u_temp, v_temp = [], []
    for i, (u, v) in enumerate(zip(U, V)):
        
        u_temp += [u]
        v_temp += [v]
        if (i + 1) % frame_interval == 0:
            # longitud de los arrays sin contar los nans:
            Nu = np.nansum(np.array(u_temp)*0 + 1, axis = 0)  
            Nv = np.nansum(np.array(v_temp)*0 + 1, axis = 0)  
            # promedio sin contar los nans:
            U_avg.append( np.nanmean(u_temp, axis=0) )
            V_avg.append( np.nanmean(v_temp, axis=0) )
            # desviacion estandar sin contar los nans:
            U_err.append( np.nanstd(u_temp, axis=0) / np.sqrt(Nu) )
            V_err.append( np.nanstd(v_temp, axis=0) / np.sqrt(Nv)  )
            # tiempo medio del intervalo:
            T_avg.append( (T[i] + T[i + 1 - frame_interval]) / 2 )

            u_temp, v_temp = [], []

    U_avg, V_avg = np.array(U_avg), np.array(V_avg) 
    U_err, V_err = np.array(U_err), np.array(V_err)
    
    return T_avg, x, y, U_avg, V_avg, U_err, V_err

def get_center(x, y, u, v, percentile = 0.1, remove_outliers = True, GRAFICAR = False):
    """
    Calcula el centro del vórtice calculando las intersecciones de las rectas perpendiculares al las velocidades de mayor magnitud, y promediandolos. Si `remove_outliers = True`, solo toma las intersecciones dentro del circulo que encierra el campo.
    x, y, u, v : ndarrys
        datos del campo en un instante (sin dimension temporal).
    percentile : float
        el percentil de velocidades que utiliza para realizar el promedio. `percentile = 1` toma todas las velocidades, `percentile = 0.1` toma el 10% de las velocidades, seleccionando las de mayor magnitúd.
    remove_outliers : bool
        si `remove_oultiers = True` solo se consideran las intersecciones dentro del circulo que encierra al campo, y se descartan las intersecciones a más de 3 sigmas del promedio.
    GRAFICAR : bool
        si mostrar o no un gráfico con todas las intersecciones. 
    """
    # Tirar las velocidades nan:
    is_nan = np.logical_or(np.isnan(u), np.isnan(v))
    x, y = x[~is_nan], y[~is_nan]
    u, v = u[~is_nan], v[~is_nan]

    # Tirar las velocidades nulas:
    is_null_velocity = np.logical_or(np.abs(u) == 0, np.abs(v) == 0)
    x, y = x[~is_null_velocity], y[~is_null_velocity]
    u, v = u[~is_null_velocity], v[~is_null_velocity]

    # Cantidad de puntos para usar:
    n = int(len(x) * percentile)
    
    # Ordenar según el módulo de la velocidad y quedarnos con las más altas:
    filtered_data = np.array( sorted( zip(x, y, u, v), key = lambda row: np.hypot(row[2], row[3]), reverse = True))[:n]
    x, y, u, v = filtered_data.T

    m = - u / v     # Pendientes de la rectas normales a la velocidad
    b = y - m * x   # Ordenadas al origen

    # Calcular la intersección de cada par posible de rectas:
    x_intercepts = []
    y_intercepts = []
    for i in range(n):
        m1, b1 = m[i], b[i]
        for j in range(i+1, n):
            m2, b2 = m[j], b[j]
            if m1 != m2:
                x_intercept = (b2 - b1) / (m1 - m2)
                y_intercept = x_intercept * m1 + b1
                x_intercepts.append(x_intercept)
                y_intercepts.append(y_intercept)

    if GRAFICAR: 
        plt.figure(figsize = (3, 3))
        ax = plt.axes()
        X_lin = np.linspace(np.min(x), np.max(x), 10) # vector de valores de x para graficar
    
        for i in range(len(x)):
            ax.plot(X_lin, m[i]*X_lin + b[i], c='k', lw=.5, alpha=.5)
            
        ax.scatter(x_intercepts, y_intercepts, c='r', s=5, alpha=.4, zorder=2)
        ax.quiver(x, y, u, v, color = 'k', edgecolor= 'k', linewidth=.3, zorder=3, scale = 200)
        ax.set_xlim((np.min(x), np.max(x)))
        ax.set_ylim((np.min(y), np.max(y)))
        ax.set_aspect('equal')
        plt.show()
    
    x_intercepts = np.array(x_intercepts)
    y_intercepts = np.array(y_intercepts)

    # Sacar intersecciones fuera del círculo que encierra a los valores de x e y:
    if remove_outliers:
        x_midpoint, y_midpoint = (np.max(x) + np.min(x)) / 2, (np.max(y) + np.min(y)) / 2

        r_limit = np.max([np.max(x - x_midpoint), np.max(y - y_midpoint)])
        r_intercepts = np.hypot(x_intercepts - x_midpoint, y_intercepts - y_midpoint)
        is_within_limit = r_intercepts < r_limit
        x_intercepts, y_intercepts = x_intercepts[is_within_limit], y_intercepts[is_within_limit]
        r_intercepts = r_intercepts[is_within_limit]

        is_outlier = np.abs(r_intercepts - np.mean(r_intercepts)) > 3*np.std(r_intercepts)
        x_intercepts = x_intercepts[~is_outlier]
        y_intercepts = y_intercepts[~is_outlier]

    # Calculo del centro como el punto medio de todas las intersecciones:
    xc = np.mean(x_intercepts)
    yc = np.mean(y_intercepts)
    err_xc = np.std(x_intercepts) / np.sqrt(len(x_intercepts))
    err_yc = np.std(y_intercepts) / np.sqrt(len(y_intercepts))
    
    return (xc, yc), (err_xc, err_yc)

def get_velocity_in_polar_coords(x, y, u, v, u_err, v_err, center, num_bins = None):
    """
    x, y, u, v, u_err, v_err : ndarrays
        datos del campo del vórtice en coordenadas cartesianas.
    center : tuple
        tupla con las coordenadas (x, y) del centro del vórtice.
    num_bins : int
        número de bins para dividir los datos radiales.

    DEVUELVE: 
    r : ndarray
        array con los radios.
    vr, vt : ndarray
        arrays con los componentes de velocidad radial y tangencial.
    vr_err, vt_err
        array con los errores en los componentes de velocidad radial y tangencial.

    """
    
    # Cambia el origen:
    x, y = x - center[0], y - center[1]

    # Tira las velocidades nan:
    is_nan = np.logical_or(np.isnan(u), np.isnan(v))
    x, y = x[~is_nan], y[~is_nan]
    u, v = u[~is_nan], v[~is_nan]
    u_err, v_err = u_err[~is_nan], v_err[~is_nan]

    # Tira los puntos del origen:
    is_vortex_center = np.hypot(x, y) == 0
    x, y = x[~is_vortex_center], y[~is_vortex_center]
    u, v = u[~is_vortex_center], v[~is_vortex_center]
    u_err, v_err = u_err[~is_vortex_center], v_err[~is_vortex_center]

    # Tira las velocidades nulas:
    is_null_velocity = np.logical_or(np.abs(u) == 0, np.abs(v) == 0)
    x, y = x[~is_null_velocity], y[~is_null_velocity]
    u, v = u[~is_null_velocity], v[~is_null_velocity]
    u_err, v_err = u_err[~is_null_velocity], v_err[~is_null_velocity]

    # Pasa a polares:
    r  = np.hypot(x, y)
    th = np.angle(x + 1j*y)

    # Calcula velocidad tangencial y radial:
    vr =  np.cos(th)*u + np.sin(th)*v
    vt = -np.sin(th)*u + np.cos(th)*v
    vr_err = np.sqrt((np.cos(th)*u_err)**2 + (np.sin(th)*v_err)**2)
    vt_err = np.sqrt((np.sin(th)*u_err)**2 + (np.cos(th)*v_err)**2)

    # Ordena por radio:
    r, th, vr, vt, vr_err, vt_err = np.array(sorted(zip(r, th, vr, vt, vr_err, vt_err), key = lambda row: row[0])).T

    if num_bins is not None:
        # Bines de radio (franjas donde se va a promediar):
        bin_edges   = np.linspace(min(r), max(r), num_bins + 1)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        # Calcular el promedio de las velocidades tangenciales en cada bin:
        vt_avgs, vt_avgs_err = [], []
        vr_avgs, vr_avgs_err = [], []
        r_avgs = []
        for i in range(num_bins):
            is_in_bin = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
            vr_selection = vr[is_in_bin]
            vt_selection = vt[is_in_bin]
            vr_err_selection = vr_err[is_in_bin]
            vt_err_selection = vt_err[is_in_bin]

            if vt_selection.size > 0 and vr_selection.size > 0:

                vr_avg = np.mean(vr_selection)
                vr_avg_err = np.sqrt( ( np.sum(vr_err_selection**2) + np.var(vr_selection) ) / vr_selection.size )
                vr_avgs.append(vr_avg)
                vr_avgs_err.append(vr_avg_err)

                vt_avg = np.mean(vt_selection)
                vt_avg_err = np.sqrt( ( np.sum(vt_err_selection**2) + np.var(vt_selection) ) / vt_selection.size )
                vt_avgs.append(vt_avg)
                vt_avgs_err.append(vt_avg_err)

                r_avgs.append(bin_centers[i])

        r, vr, vt, vr_err, vt_err = np.array(r_avgs), np.array(vr_avgs), np.array(vt_avgs), np.array(vr_avgs_err), np.array(vt_avgs_err)

    return r, vr, vt, vr_err, vt_err

def cmap(u, v, colormap):
    color = np.hypot(u, v)**2
    color = (color - np.nanmin(color)) / (np.nanmax(color) - np.nanmin(color))
    return colormaps[colormap](color)