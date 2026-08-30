"""
Métricas derivadas de una serie de frecuencia cardíaca.

Contiene funciones puras: reciben arreglos de tiempo y pulso y devuelven
indicadores, sin leer ni escribir datos. Vive en su propio módulo porque tanto
la capa de adquisición como la de procesamiento necesitan estos cálculos, y
alojarlos en cualquiera de las dos crearía una dependencia circular entre ambas.
"""

import numpy as np

# Modelo de cinco zonas por porcentaje de la frecuencia cardíaca máxima. El cuarto
# valor es el multiplicador del TRIMP de Edwards (1993): un minuto en zona 5 aporta
# cinco veces más carga que uno en zona 1.
ZONAS = [
    ("Z1", 0.50, 0.60, 1),
    ("Z2", 0.60, 0.70, 2),
    ("Z3", 0.70, 0.80, 3),
    ("Z4", 0.80, 0.90, 4),
    ("Z5", 0.90, 1.01, 5),
]

# Por encima de esta variabilidad relativa el esfuerzo deja de considerarse estable
# y la deriva cardíaca no es interpretable. Es un umbral heurístico, no un estándar.
UMBRAL_VARIABILIDAD = 8.0


def _intervalos(tiempos, pulsos):
    """Devuelve la duración de cada tramo y su pulso medio, por regla del trapecio."""
    if len(tiempos) < 2:
        return np.array([]), np.array([])
    return np.diff(tiempos), (pulsos[:-1] + pulsos[1:]) / 2


def media_ponderada_por_tiempo(tiempos, pulsos):
    """Integra el pulso por regla del trapecio y lo divide entre la duración total.

    El trazado manual produce muestras a intervalos irregulares, así que un promedio
    simple sobrepondera los tramos donde el dedo avanzó más lento.
    """
    if len(tiempos) == 0:
        return None
    if len(tiempos) == 1:
        return float(pulsos[0])
    duraciones, medios = _intervalos(tiempos, pulsos)
    total = duraciones.sum()
    if total <= 0:
        return float(np.mean(pulsos))
    return float((medios * duraciones).sum() / total)


def tiempo_en_zonas(tiempos, pulsos, fc_maxima):
    """Reparte la duración de la actividad entre las cinco zonas de intensidad.

    El tiempo por debajo del 50 % de la frecuencia máxima queda fuera del reparto,
    tal como define el modelo, así que la suma de las zonas puede ser menor que la
    duración total de la sesión.

    Args:
        tiempos (np.ndarray): Segundos transcurridos en cada muestra.
        pulsos (np.ndarray): Pulsaciones registradas.
        fc_maxima (float): Referencia de frecuencia cardíaca máxima del atleta.
    Returns:
        dict[str, float]: Segundos acumulados en cada zona.
    """
    reparto = {nombre: 0.0 for nombre, _, _, _ in ZONAS}
    duraciones, medios = _intervalos(tiempos, pulsos)
    if duraciones.size == 0 or not fc_maxima:
        return reparto

    proporciones = medios / float(fc_maxima)
    for nombre, bajo, alto, _ in ZONAS:
        en_zona = (proporciones >= bajo) & (proporciones < alto)
        reparto[nombre] = float(duraciones[en_zona].sum())
    return reparto


def trimp_edwards(tiempos, pulsos, fc_maxima):
    """Suma los minutos de cada zona ponderados por su multiplicador de intensidad."""
    reparto = tiempo_en_zonas(tiempos, pulsos, fc_maxima)
    return sum((reparto[nombre] / 60) * peso for nombre, _, _, peso in ZONAS)


def deriva_cardiaca(tiempos, pulsos):
    """Compara el pulso medio de la segunda mitad de la sesión con el de la primera.

    Un pulso que sube manteniendo el mismo esfuerzo indica fatiga o deshidratación.
    Solo tiene sentido en esfuerzos continuos: en series o cuestas repetidas la
    oscilación propia del entrenamiento domina sobre la deriva real, por lo que se
    devuelve además una marca de estabilidad.

    Returns:
        dict: Porcentaje de deriva, variabilidad relativa y si el esfuerzo fue estable.
    """
    vacio = {"deriva_pct": None, "variabilidad_pct": None, "estable": False}
    if len(tiempos) < 4:
        return vacio

    corte = (float(tiempos[0]) + float(tiempos[-1])) / 2
    primera = tiempos <= corte
    segunda = tiempos >= corte
    if primera.sum() < 2 or segunda.sum() < 2:
        return vacio

    media_inicial = media_ponderada_por_tiempo(tiempos[primera], pulsos[primera])
    media_final = media_ponderada_por_tiempo(tiempos[segunda], pulsos[segunda])
    if not media_inicial:
        return vacio

    promedio = float(np.mean(pulsos))
    variabilidad = 100 * float(np.std(pulsos)) / promedio if promedio else 0.0
    return {
        "deriva_pct": 100 * (media_final - media_inicial) / media_inicial,
        "variabilidad_pct": variabilidad,
        "estable": variabilidad <= UMBRAL_VARIABILIDAD,
    }


def resumen_serie(tiempos, pulsos, fc_maxima):
    """Agrupa en un solo diccionario todos los indicadores de una serie."""
    return {
        "muestras": len(tiempos),
        "duracion_s": float(tiempos[-1] - tiempos[0]) if len(tiempos) > 1 else 0.0,
        "fc_media": media_ponderada_por_tiempo(tiempos, pulsos),
        "fc_maxima": float(np.max(pulsos)) if len(pulsos) else None,
        "zonas": tiempo_en_zonas(tiempos, pulsos, fc_maxima),
        "trimp_edwards": trimp_edwards(tiempos, pulsos, fc_maxima),
        "deriva": deriva_cardiaca(tiempos, pulsos),
    }
