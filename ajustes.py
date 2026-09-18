"""Ajustes personales del atleta: lectura, escritura y valores derivados.

Los ajustes viven en Supabase y no en session_state porque session_state se
reinicia cada vez que se abre la aplicación.
"""
from datetime import date

import streamlit as st

import config
import data_loader
import metricas_fc

# Valores de respaldo. Solo entran en juego si la base de datos no responde, para
# que la aplicación siga en pie en lugar de caerse por un fallo de red.
RESPALDO = {
    "fecha_nacimiento": None,
    "fc_maxima_medida": None,
    "fc_reposo": None,
    "modelo_zonas": "fcmax",
    "deporte_defecto": "Carrera",
    "periodo_defecto": "Todo el histórico",
    "actividades_agente": 20,
    "retencion_diario": 30,
}

# Se centraliza aquí porque el origen de la FC máxima se muestra en dos pantallas y
# tenerlo duplicado acabaría con una diciendo una cosa y la otra diciendo otra.
ORIGEN_FCMAX = {
    "medida": "medida por ti",
    "tanaka": "estimada por tu edad",
    "respaldo": "de respaldo del código",
}

def completar(guardados):
    """Rellena con los valores de respaldo los campos que la base de datos no traiga.

    Args:
        guardados (dict | None): Fila leída de Supabase, o None si la lectura falló.
    Returns:
        dict: Ajustes completos, con la fecha de nacimiento ya convertida a date.
    """
    ajustes = dict(RESPALDO)
    if guardados:
        for clave in RESPALDO:
            if guardados.get(clave) is not None:
                ajustes[clave] = guardados[clave]

    if isinstance(ajustes["fecha_nacimiento"], str):
        ajustes["fecha_nacimiento"] = date.fromisoformat(ajustes["fecha_nacimiento"])
    return ajustes


@st.cache_data(ttl=600)
def leer():
    """Devuelve los ajustes vigentes, completados con los valores de respaldo."""
    return completar(data_loader.leer_ajustes())

def guardar(valores):
    """Escribe los ajustes y descarta la caché para que la aplicación los vea al instante.

    Args:
        valores (dict): Campos a modificar, con los nombres de columna de la tabla.
    Returns:
        bool: True si la escritura llegó a la base de datos.
    """
    datos = dict(valores)
    if isinstance(datos.get("fecha_nacimiento"), date):
        datos["fecha_nacimiento"] = datos["fecha_nacimiento"].isoformat()

    if data_loader.guardar_ajustes(datos):
        leer.clear()
        return True
    return False

def edad(ajustes, referencia=None):
    """Calcula la edad cumplida, o None si no hay fecha de nacimiento guardada."""
    nacimiento = ajustes.get("fecha_nacimiento")
    if nacimiento is None:
        return None

    referencia = referencia or config.hoy()
    # Resta un año si el cumpleaños de este año todavía no ha llegado.
    cumplio = (referencia.month, referencia.day) >= (nacimiento.month, nacimiento.day)
    return referencia.year - nacimiento.year - (0 if cumplio else 1)

def fc_maxima_tanaka(ajustes):
    """Estima la FC máxima con Tanaka et al. (2001), o None sin fecha de nacimiento."""
    años = edad(ajustes)
    return None if años is None else round(208 - 0.7 * años)

def fc_maxima_efectiva(ajustes):
    """Devuelve la FC máxima que debe usar la aplicación y de dónde sale.

    Returns:
        tuple[int, str]: Pulsaciones y origen del valor, 'medida', 'tanaka' o
            'respaldo', para que la interfaz pueda dejarlo claro.
    """
    if ajustes.get("fc_maxima_medida"):
        return int(ajustes["fc_maxima_medida"]), "medida"

    estimada = fc_maxima_tanaka(ajustes)
    if estimada:
        return estimada, "tanaka"
    return config.FC_MAXIMA, "respaldo"

def usa_karvonen(ajustes):
    """Indica si toca aplicar Karvonen, que exige tener la FC en reposo guardada."""
    return ajustes.get("modelo_zonas") == "karvonen" and bool(ajustes.get("fc_reposo"))

def rangos_de_zonas(ajustes):
    """Devuelve las cinco zonas en pulsaciones según el modelo configurado.

    Returns:
        list[tuple[str, float, float]]: Nombre, pulso inicial y pulso final de cada
            zona. El final de la Z5 es infinito porque esa zona no tiene tope.
    """
    fc_maxima, _ = fc_maxima_efectiva(ajustes)
    reposo = int(ajustes["fc_reposo"]) if usa_karvonen(ajustes) else 0
    return metricas_fc.limites_en_pulsaciones(fc_maxima, reposo)

def indice_por_defecto(ajustes, campo, opciones):
    """Devuelve qué opción debe venir preseleccionada en un control de la barra lateral.

    El periodo se guarda como 'Año actual' en vez de un año concreto para que en
    enero la aplicación no arranque mostrando el año anterior.

    Args:
        ajustes (dict): Ajustes ya leídos.
        campo (str): 'deporte_defecto' o 'periodo_defecto'.
        opciones (list[str]): Opciones que ofrece hoy el control.
    Returns:
        int: Índice de la opción guardada, o 0 si ya no existe entre las opciones.
    """
    guardado = ajustes.get(campo)
    if guardado == "Año actual":
        guardado = str(config.hoy().year)
    return opciones.index(guardado) if guardado in opciones else 0

def texto_rango(rango):
    """Formatea un rango de zona en pulsaciones, dejando abierta la Z5 que no tiene tope."""
    _nombre, desde, hasta = rango
    if hasta == float("inf"):
        return f"{round(desde)} y más"
    return f"{round(desde)} - {round(hasta)}"