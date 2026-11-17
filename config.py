"""
Configuración compartida del SII Athletix.

Centraliza los valores que necesitan más de un módulo. Contiene la FC máxima de
respaldo y el manejo de la fecha: tanto el backend como el agente deben razonar
sobre el día del atleta, y no sobre el del servidor donde corre la aplicación.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

# FC máxima medida con pulsómetro. La aplicación prioriza la guardada en Ajustes, luego
# Tanaka por edad, y este valor solo entra si Ajustes no tiene ninguna de las dos.
FC_MAXIMA = 199

# Streamlit Community Cloud ejecuta en UTC. Sin anclar la zona, entre las 19:00 y la
# medianoche de Ecuador el servidor ya está en el día siguiente y las ventanas móviles
# de 7 y 28 días se desplazan un día. En local nunca se reproduce.
ZONA_HORARIA = ZoneInfo("America/Guayaquil")

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

# Funciones de utilidad para el manejo de fechas
# Esta función se usa en el backend y en el agente, y no depende de la zona del servidor de Streamlit, 
# sino de la del atleta.
def hoy():
    """Devuelve la fecha actual en la zona del atleta, no la del servidor."""
    return datetime.now(ZONA_HORARIA).date()

# Esta función se usa en el backend y en el agente, y no depende del sistema operativo.
def fecha_en_texto(fecha):
    """Formatea una fecha en español sin depender del locale del sistema operativo."""
    return f"{DIAS[fecha.weekday()]} {fecha.day} de {MESES[fecha.month - 1]} de {fecha.year}"

def fecha_hora_corta(clave):
    """Convierte la clave de una actividad en la fecha y hora que se muestran en pantalla.

    Args:
        clave (str): Hora de inicio tal como la guarda Strava, por ejemplo 2026-11-16T16:00:08Z.
    Returns:
        str: Fecha y hora como 16/11/2026 16:00, o la clave tal cual si no se puede leer.
    """
    # Strava etiqueta la hora local con sufijo Z, así que se muestra sin convertir de zona.
    # La Z se reemplaza porque las versiones de Python anteriores a la 3.11 no la leen.
    try:
        return datetime.fromisoformat(str(clave).replace("Z", "+00:00")).strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return str(clave)