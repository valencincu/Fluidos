from os import path, sep, rename, makedirs, listdir

def crear_carpeta_video(analysis_dir: str, video_source: str, video_file: str):
    """
    Una función que toma un video de la fuente de videos `video_source`, y crea una carpeta en `analysis_dir` \
    con el mismo nombre del video. Luego mueve al video a esta carpeta para poder analizarlo ahí. 
    analysis_dir : str
        carpeta donde se van a encontrar las carpetas correspondientes a cada video.
    video_source : str
        carpeta donde se encuentran todos los videos que no tienen carpeta propia.
    video_file : str
        Video que se encuentra en `video_source` para el cual se va a crear la carpeta. E.g. 'mi_video.mp4'
    """
    id = path.splitext(video_file)[0]    # Toma como identificación del video su nombre sin la extensión
    new_dir =  analysis_dir + sep + id   # Nueva carpeta del video
    if not path.exists(video_source + sep + video_file) and not path.exists(new_dir + sep + video_file):
        raise Exception(f"No existe un video llamado '{video_file}' en '{video_source}'")
    if not path.exists(new_dir):
        makedirs(new_dir)
        f = open(new_dir + sep + id + ".txt", "w")
        rename(video_source + sep + video_file, new_dir + sep + video_file)
        print(f"Se creo la carpeta '{id}' en {analysis_dir}.")
    else: 
        print(f"Ya existe una carpeta con la identificación de '{video_file}'. No se creo una carpeta nueva con esta identificación.")
        