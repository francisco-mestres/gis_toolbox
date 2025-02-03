import os
import logging
from tqdm import tqdm


from gis_toolbox.enums import OSType, GpdEngine


####### EINSTELLUNGEN #######

class TqdmConfig:
    DEFAULT_UNIT_CHUNKS = "rows"


class LoggingMessages:
    DEFAULT_WRITING_CHUNKS = "Writing chunks to {file}"


class Config:
    DEFAULT_CHUNK_SIZE = 10_000
    DEFAULT_DRIVER = 'ESRI Shapefile'
    DEFAULT_IO_ENGINE = GpdEngine.PYOGRIO.value
    DEFAULT_URL_PATTERN = 'https://example.com/{x_key}/{y_key}/{spacing}.tif'


####### HAUPTFUNKTIONEN #######


def write_gdf_to_file(gdf, file_path, logging_message, engine=Config.DEFAULT_IO_ENGINE):

    gdf.to_file(file_path, driver=Config.DEFAULT_DRIVER, engine=engine)
    logging.info(logging_message.format(file=file_path))


def write_gdf_in_chunks(gdf, 
                        output_path, 
                        chunk_size=Config.DEFAULT_CHUNK_SIZE, 
                        driver=Config.DEFAULT_DRIVER, 
                        desc_message=LoggingMessages.DEFAULT_WRITING_CHUNKS,
                        engine=Config.DEFAULT_IO_ENGINE):
    
    total_rows = len(gdf)
    for i in tqdm(range(0, total_rows, chunk_size), 
                  desc=desc_message, 
                  unit=TqdmConfig.DEFAULT_UNIT_CHUNKS):

        chunk = gdf.iloc[i:i + chunk_size]
        if i == 0:
            chunk.to_file(output_path, driver=driver, engine=engine)
        else:
            chunk.to_file(output_path, driver=driver, mode='a', engine=engine)


def ensure_folder_exists(folder_path, logging_message):

    if not os.path.exists(folder_path):
        logging.info(logging_message.format(folder=folder_path))
        os.makedirs(folder_path)


####### PFADVERWALTUNG #######

import re


def convert_path_to_unix(path):
    """Konvertiert einen Windows-Pfad in einen Unix-Pfad."""
    drive_match = re.match(r"([A-Za-z]):\\", path)
    if drive_match:
        drive_letter = drive_match.group(1).lower()
        return path.replace(f"{drive_letter.upper()}:\\", f"/mnt/{drive_letter}/").replace("\\", "/")
    return path


def convert_path_to_windows(path):
    """Konvertiert einen Unix-Pfad in einen Windows-Pfad."""
    if path.startswith("/mnt/"):
        drive_letter = path[5].upper()
        return path.replace(f"/mnt/{path[5]}/", f"{drive_letter}:/").replace("/", "\\")
    return path


def is_unix_path(path):
    """Prüft, ob ein Pfad im Unix-Format vorliegt."""
    return path.startswith("/mnt/")


def is_windows_path(path):
    """Prüft, ob ein Pfad im Windows-Format vorliegt."""
    return re.match(r"[A-Za-z]:\\", path) is not None


def adapt_path(path):
    """Konvertiert einen Pfad basierend auf dem Betriebssystem."""
    system_os = os.name

    if system_os == OSType.WINDOWS and is_unix_path(path):
        return convert_path_to_windows(path)
    
    elif system_os == OSType.UNIX and is_windows_path(path):
        return convert_path_to_unix(path)
    
    return path
