"""
Módulo Predictivo del SII Athletix.

Implementa una regresión simple entrenada sobre promedios consolidados por
mesociclo (bloques de 28 días), evitando así el sesgo del tapering que
introduciría un análisis semanal previo a competición.

Incluye:
    - Entrenamiento y validación cruzada Leave-One-Out del modelo.
    - Conversión del ritmo predicho a cualquier distancia mediante Riegel.
    - Verificador de viabilidad: contrasta una meta contra la progresión
      histórica real del atleta, sin proyectar un volumen de entrenamiento.

Nota metodológica, registrada tras comparar seis variantes con validación
cruzada Leave-One-Out sobre el mismo conjunto de mesociclos:

El modelo original usaba tres variables, entre ellas el ritmo medio de las
carreras del propio bloque para predecir el mejor ritmo de ese bloque. Esa
variable predice el mínimo de un conjunto a partir del promedio de ese mismo
conjunto, lo cual es aritmética y no una relación de entrenamiento. Producía
un R² de 0.78 que no sobrevivió a la comparación honesta.

Se probaron seis variantes contra dos referencias, "repetir la media
histórica" y "repetir la marca del bloque anterior". El volumen semanal y las
horas de entrenamiento, solas o combinadas, no superaron ni la referencia más
simple: en el historial disponible no hay variación suficiente de volumen
entre bloques como para que su efecto se pueda medir, aunque eso no implica
que el volumen sea irrelevante en la realidad. La única variable que superó
ambas referencias fue el mejor ritmo del bloque anterior, con MAE de 0.360
min/km frente a 0.535 de la referencia ingenua. Añadir volumen, horas o
desnivel a esa variable no mejoró el resultado de forma consistente.

Se intentó además calibrar un factor de metros de desnivel equivalentes a un
kilómetro llano, para incorporar las carreras de montaña. Cuatro métodos de
estimación arrojaron valores entre 100 y 1015 metros, una dispersión
demasiado amplia para ser de fiar: el historial no tiene suficiente variedad
de pendientes e intensidades para separar el efecto del terreno del efecto
del esfuerzo. Se descartó y quedó pendiente para cuando existan series GPS
punto a punto, que permitan calcular el Grade Adjusted Pace real en lugar de
promediar toda la actividad.

Por esto el módulo pasó de una regresión de tres variables a una de una sola
variable, y el planificador inverso, que despejaba el volumen semanal de una
ecuación que ya no existe, se convirtió en un verificador de viabilidad que
no proyecta kilómetros, solo contrasta la meta contra la progresión medida.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

import config

# Constantes del modelo
DISTANCIA_MINIMA_ESFUERZO = 5.0        # Km mínimos para considerar un esfuerzo válido
RANGO_RITMO_VALIDO = (3.0, 10.0)       # Ritmos plausibles de carrera en llano (min/km)
EXPONENTE_RIEGEL = 1.06                # Coeficiente estándar de fatiga por distancia
RANGO_CALIBRADO = (5, 12)              # Rango de distancias con historial suficiente
MESOCICLOS_MINIMOS = 10                # Muestra mínima para un entrenamiento honesto

CARACTERISTICAS = ["mejor_ritmo_previo"]   # Única variable que superó ambas referencias


def construir_dataset_mesociclos(df):
    """
    Transforma el historial de actividades en un dataset por mesociclo.

    Variable objetivo: el mejor ritmo (min/km) alcanzado en el bloque en carreras
    de al menos 5 km. Actúa como indicador proxy del rendimiento en competencia,
    dado que el historial no contiene carreras oficiales etiquetadas.
    """
    if df.empty or "Mesociclo" not in df.columns:
        return pd.DataFrame()

    carreras = df[
        (df["Tipo de actividad"] == "Carrera")
        & (df["Distancia_km"] >= DISTANCIA_MINIMA_ESFUERZO)
        & (df["Ritmo (min/km)"].between(*RANGO_RITMO_VALIDO))
    ]
    if carreras.empty:
        return pd.DataFrame()

    datos = pd.DataFrame(index=sorted(carreras["Mesociclo"].unique()))
    datos["mejor_ritmo"] = carreras.groupby("Mesociclo")["Ritmo (min/km)"].min()

    # El ritmo del bloque anterior solo es válido si los bloques son consecutivos:
    # un salto en el índice significa semanas sin carreras válidas de por medio, y
    # el "bloque anterior" dejaría de ser realmente el mes justo antes de este.
    datos["mejor_ritmo_previo"] = datos["mejor_ritmo"].shift(1)
    consecutivo = datos.index.to_series().diff() == 1
    datos.loc[~consecutivo.fillna(False), "mejor_ritmo_previo"] = np.nan

    return datos.dropna()


def entrenar_modelo(df):
    """
    Entrena la regresión y la valida con Leave-One-Out contra dos referencias
    ingenuas, para que el diagnóstico distinga una relación real de una que
    solo parece buena por el ajuste dentro de la propia muestra.

    Devuelve un diccionario con el modelo ajustado, sus métricas de validación
    fuera de muestra y el dataset utilizado. Devuelve None si no hay muestra
    suficiente para un entrenamiento honesto.
    """
    datos = construir_dataset_mesociclos(df)

    if len(datos) < MESOCICLOS_MINIMOS:
        return None

    X = datos[CARACTERISTICAS]
    y = datos["mejor_ritmo"]

    modelo = LinearRegression()
    predicciones_cv = cross_val_predict(modelo, X, y, cv=LeaveOneOut())
    r2 = r2_score(y, predicciones_cv)
    mae = mean_absolute_error(y, predicciones_cv)
    mae_baseline = np.abs(y - y.mean()).mean()
    # Referencia dura: repetir sin más la marca del bloque anterior. Superarla es
    # lo que demuestra que la regresión aporta algo más que copiar el último dato.
    mae_marca_previa = mean_absolute_error(y, datos["mejor_ritmo_previo"])

    modelo.fit(X, y)

    return {
        "modelo": modelo,
        "datos": datos,
        "r2": r2,
        "mae": mae,
        "mae_baseline": mae_baseline,
        "mae_marca_previa": mae_marca_previa,
        "n_mesociclos": len(datos),
        "coeficientes": dict(zip(CARACTERISTICAS, modelo.coef_)),
    }


def predecir_ritmo(entrenamiento, metricas_bloque):
    """Predice el ritmo de competición (min/km) a partir de las métricas del bloque."""
    if entrenamiento is None or metricas_bloque.get("mejor_ritmo_previo") is None:
        return np.nan

    entrada = pd.DataFrame([metricas_bloque])[CARACTERISTICAS]
    return float(entrenamiento["modelo"].predict(entrada)[0])


def metricas_ultimo_bloque(df, hoy=None, dias=28):
    """
    Extrae las métricas del mesociclo vigente para alimentar el modelo. El ritmo
    del bloque anterior se toma de los 28 días previos a la ventana vigente, no
    del dataset de entrenamiento, para reflejar la forma más reciente del atleta.

    Devuelve None si el bloque vigente o el anterior no contienen carreras válidas.
    """
    # La fecha se ancla a la zona horaria del atleta: el servidor corre en UTC y
    # de noche desplazaría un día la ventana de 28 días.
    hoy = pd.Timestamp(hoy).normalize() if hoy is not None else pd.Timestamp(config.hoy())
    inicio_vigente = hoy - pd.Timedelta(days=dias)
    inicio_previo = inicio_vigente - pd.Timedelta(days=dias)

    def _mejor_ritmo(desde, hasta):
        bloque = df[(df["Fecha"] >= desde) & (df["Fecha"] < hasta)]
        carreras = bloque[
            (bloque["Tipo de actividad"] == "Carrera")
            & (bloque["Distancia_km"] >= DISTANCIA_MINIMA_ESFUERZO)
            & (bloque["Ritmo (min/km)"].between(*RANGO_RITMO_VALIDO))
        ]
        return carreras["Ritmo (min/km)"].min() if not carreras.empty else None

    mejor_ritmo_previo = _mejor_ritmo(inicio_previo, inicio_vigente)
    if mejor_ritmo_previo is None:
        return None

    return {"mejor_ritmo_previo": float(mejor_ritmo_previo)}


def ritmo_a_tiempo(ritmo_min_km, distancia_km, distancia_referencia=10.0):
    """
    Convierte un ritmo de referencia al tiempo total de una distancia objetivo
    aplicando la fórmula de Riegel, que corrige la degradación fisiológica del
    ritmo conforme aumenta la distancia (multiplicar linealmente sobreestima
    el rendimiento en pruebas largas).
    """
    if ritmo_min_km is None or np.isnan(ritmo_min_km):
        return np.nan

    tiempo_referencia = ritmo_min_km * distancia_referencia
    factor = (distancia_km / distancia_referencia) ** EXPONENTE_RIEGEL
    return tiempo_referencia * factor


def formatear_tiempo(minutos_decimales):
    """Transforma minutos decimales a formato de cronómetro (h:mm:ss)."""
    if minutos_decimales is None or np.isnan(minutos_decimales):
        return "—"

    total_segundos = int(round(minutos_decimales * 60))
    horas, resto = divmod(total_segundos, 3600)
    minutos, segundos = divmod(resto, 60)

    if horas:
        return f"{horas}h {minutos:02d}m {segundos:02d}s"
    return f"{minutos}m {segundos:02d}s"


def fuera_de_rango_calibrado(distancia_km):
    """Indica si la distancia objetivo excede el rango con el que se entrenó el modelo."""
    return not (RANGO_CALIBRADO[0] <= distancia_km <= RANGO_CALIBRADO[1])


def progresion_historica(datos, meses):
    """Mide cuánto suele mejorar el atleta en una ventana de una cantidad de meses.

    Compara, para cada par de bloques separados por esa cantidad de mesociclos, la
    diferencia real de marca, y devuelve la mejora típica en ese plazo. Es el dato
    con el que se contrasta si una meta es ambiciosa: no una opinión, sino lo que el
    propio historial del atleta ha logrado en plazos comparables.

    Args:
        datos (pd.DataFrame): Salida de construir_dataset_mesociclos.
        meses (int): Cantidad de mesociclos de separación a comparar (1 mes = 1 bloque).
    Returns:
        dict | None: Mejora media en min/km y cuántos pares la sostienen, o None si
            no hay al menos tres pares para no ofrecer una cifra sin respaldo.
    """
    serie = datos["mejor_ritmo"]
    diferencias = (serie.shift(meses) - serie).dropna()
    if len(diferencias) < 3:
        return None
    return {"mejora_media_min_km": float(diferencias.mean()), "pares": int(len(diferencias))}


def verificar_viabilidad(entrenamiento, metricas_actuales, tiempo_objetivo_min, distancia_km):
    """
    Contrasta una meta de carrera contra el ritmo estimado y la progresión histórica
    del atleta. No proyecta un volumen de entrenamiento porque el historial no
    sostiene esa relación: ver la nota metodológica al inicio del módulo.

    Args:
        entrenamiento (dict | None): Salida de entrenar_modelo.
        metricas_actuales (dict | None): Salida de metricas_ultimo_bloque.
        tiempo_objetivo_min (float): Meta en minutos, para la distancia dada.
        distancia_km (float): Distancia de la meta.
    Returns:
        dict: Diagnóstico con el ritmo objetivo, el margen de error del modelo y
            cómo se compara la mejora exigida contra la progresión típica del atleta.
    """
    if entrenamiento is None or metricas_actuales is None:
        return {"viable": False,
                "mensaje": "No hay datos recientes suficientes para comparar la meta."}

    ritmo_objetivo_ref = (tiempo_objetivo_min
                          / ((distancia_km / 10.0) ** EXPONENTE_RIEGEL)) / 10.0
    ritmo_actual = predecir_ritmo(entrenamiento, metricas_actuales)
    mae = entrenamiento["mae"]

    mejora_exigida = ritmo_actual - ritmo_objetivo_ref
    mejora_pct = 100 * mejora_exigida / ritmo_actual if ritmo_actual else 0.0

    resultado = {
        "viable": True,
        "ritmo_objetivo": ritmo_objetivo_ref,
        "ritmo_actual": ritmo_actual,
        "margen_error": mae,
        "mejora_exigida_min_km": mejora_exigida,
        "mejora_exigida_pct": mejora_pct,
        "referencias": {},
    }

    # Se contrasta contra 1, 2 y 3 mesociclos porque ahí se concentran las metas a
    # corto y mediano plazo; a más meses la muestra de pares se vuelve demasiado chica.
    for meses in (1, 2, 3):
        progresion = progresion_historica(entrenamiento["datos"], meses)
        if progresion:
            resultado["referencias"][meses] = progresion

    return resultado
