"""
Configuración compartida del SII Athletix.

Centraliza los valores que necesitan más de un módulo. Hoy solo contiene el
manejo de la fecha: tanto el backend como el agente deben razonar sobre el día
del atleta, y no sobre el del servidor donde corre la aplicación.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

# Frecuencia cardíaca máxima estimada con la fórmula de Tanaka et al. (2001),
# FCmax = 208 - 0.7 x edad, más ajustada que 220 - edad en personas entrenadas.
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