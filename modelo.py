"""
Módulo Predictivo del SII Athletix.

Implementa un modelo de Regresión Lineal Múltiple entrenado sobre promedios
consolidados por mesociclo (bloques de 28 días), evitando así el sesgo del
tapering que introduciría un análisis semanal previo a competición.

Incluye:
    - Entrenamiento y validación cruzada Leave-One-Out del modelo.
    - Conversión del ritmo predicho a cualquier distancia mediante Riegel.
    - Planificador inverso: despeja el volumen necesario para una meta dada.
"""

import numpy as np                                                                 # Operaciones numéricas
import pandas as pd                                                                # Manejo de datos tabulares
import config
from sklearn.linear_model import LinearRegression                                  # Modelo de regresión lineal múltiple
from sklearn.model_selection import LeaveOneOut, cross_val_predict                 # Validación cruzada exhaustiva
from sklearn.metrics import r2_score, mean_absolute_error                          # Métricas de error del modelo

# Constantes del modelo
DISTANCIA_MINIMA_ESFUERZO = 5.0                                                    # Km mínimos para considerar un esfuerzo válido
RANGO_RITMO_VALIDO = (3.0, 10.0)                                                   # Ritmos plausibles de carrera (min/km)
EXPONENTE_RIEGEL = 1.06                                                            # Coeficiente estándar de fatiga por distancia
RANGO_CALIBRADO = (5, 12)                                                          # Rango de distancias con historial suficiente

CARACTERISTICAS = [                                                                # Variables independientes del modelo
    "ritmo_medio_mesociclo",                                                       # Ritmo medio de las carreras del bloque
    "km_semana_mesociclo",                                                         # Volumen semanal medio del bloque
    "horas_semana_mesociclo",                                                      # Horas de entrenamiento total por semana
]

# Nota metodológica: el número de sesiones se descartó como variable independiente.
# Su correlación con el volumen semanal es aprox de 0.97 (puede variar según como 
# sea la línea de tiempo de los datos), y esa colinealidad invertía el
# signo del coeficiente del volumen (el modelo concluía que correr más kilómetros
# te hace más lento), lo que impedía usar la ecuación de forma inversa en el
# planificador. Al retirarla, el R² apenas cae de 0.80 a 0.78 y el modelo recupera
# la coherencia física.

def construir_dataset_mesociclos(df):
    """
    Transforma el historial de actividades en un dataset por mesociclo.

    Variable objetivo: el mejor ritmo (min/km) alcanzado en el bloque en carreras
    de al menos 5 km. Actúa como indicador proxy del rendimiento en competencia,
    dado que el historial no contiene carreras oficiales etiquetadas.
    """
    if df.empty or "Mesociclo" not in df.columns:                                  # Sin datos no hay dataset posible
        return pd.DataFrame()                                                      # Devuelve un DataFrame vacío

    carreras = df[                                                                 # Filtra los esfuerzos válidos a pie
        (df["Tipo de actividad"] == "Carrera")
        & (df["Distancia_km"] >= DISTANCIA_MINIMA_ESFUERZO)
        & (df["Ritmo (min/km)"].between(*RANGO_RITMO_VALIDO))
    ]
    if carreras.empty:                                                             # Sin carreras no hay variable objetivo
        return pd.DataFrame()                                                      # Devuelve un DataFrame vacío

    resistencia = df[df["Tipo de actividad"].isin(["Carrera", "Bicicleta"])]       # Base aeróbica global del deportista

    datos = pd.DataFrame(index=sorted(carreras["Mesociclo"].unique()))             # Un registro por mesociclo con carreras
    datos["ritmo_medio_mesociclo"] = carreras.groupby("Mesociclo")["Ritmo (min/km)"].mean()  # Ritmo medio del bloque
    datos["km_semana_mesociclo"] = carreras.groupby("Mesociclo")["Distancia_km"].sum() / 4   # Volumen semanal de carrera
    datos["sesiones_mesociclo"] = carreras.groupby("Mesociclo").size()             # Frecuencia de carrera en el bloque
    datos["horas_semana_mesociclo"] = resistencia.groupby("Mesociclo")["Minutos"].sum() / 4 / 60  # Horas aeróbicas semanales
    datos["mejor_ritmo"] = carreras.groupby("Mesociclo")["Ritmo (min/km)"].min()   # Variable objetivo del modelo

    datos["horas_semana_mesociclo"] = datos["horas_semana_mesociclo"].fillna(0)    # Bloques sin base aeróbica registrada
    datos = datos.dropna()                                                         # Descarta bloques incompletos

    return datos                                                                   # Devuelve el dataset consolidado

def entrenar_modelo(df):
    """
    Entrena la Regresión Lineal Múltiple y la valida con Leave-One-Out.

    Devuelve un diccionario con el modelo ajustado, sus métricas de validación
    fuera de muestra y el dataset utilizado. Devuelve None si no hay muestra
    suficiente para un entrenamiento honesto.
    """
    datos = construir_dataset_mesociclos(df)                                       # Consolida el historial por bloques

    if len(datos) < 10:                                                            # Muestra insuficiente para modelar
        return None                                                                # Evita entregar un modelo no fiable

    X = datos[CARACTERISTICAS]                                                     # Matriz de variables independientes
    y = datos["mejor_ritmo"]                                                       # Vector de la variable dependiente

    modelo = LinearRegression()                                                    # Instancia el algoritmo de regresión
    predicciones_cv = cross_val_predict(modelo, X, y, cv=LeaveOneOut())            # Predice cada bloque sin haberlo visto
    r2 = r2_score(y, predicciones_cv)                                              # Coeficiente de determinación real
    mae = mean_absolute_error(y, predicciones_cv)                                  # Error absoluto medio en min/km
    mae_baseline = np.abs(y - y.mean()).mean()                                     # Error de predecir siempre la media

    modelo.fit(X, y)                                                               # Ajuste final con todos los mesociclos

    return {                                                                       # Empaqueta el resultado del entrenamiento
        "modelo": modelo,                                                          # Modelo entrenado listo para predecir
        "datos": datos,                                                            # Dataset de mesociclos utilizado
        "r2": r2,                                                                  # Bondad de ajuste fuera de muestra
        "mae": mae,                                                                # Error medio en minutos por kilómetro
        "mae_baseline": mae_baseline,                                              # Referencia contra la que se compara
        "n_mesociclos": len(datos),                                                # Tamaño real de la muestra de entrenamiento
        "coeficientes": dict(zip(CARACTERISTICAS, modelo.coef_)),                  # Peso de cada variable independiente
    }

def predecir_ritmo(entrenamiento, metricas_bloque):
    """Predice el ritmo de competición (min/km) a partir de las métricas del bloque."""
    if entrenamiento is None:                                                      # Sin modelo no hay predicción
        return np.nan                                                              # Devuelve un valor no disponible

    entrada = pd.DataFrame([metricas_bloque])[CARACTERISTICAS]                     # Ordena las variables como en el ajuste
    return float(entrenamiento["modelo"].predict(entrada)[0])                      # Devuelve el ritmo estimado

def metricas_ultimo_bloque(df, hoy=None, dias=28):
    """
    Extrae las métricas del mesociclo vigente (últimos 28 días) para alimentar
    el modelo. Devuelve None si el bloque no contiene carreras válidas.
    """

    # La fecha se ancla a la zona horaria del atleta: el servidor corre en UTC y de noche
    # desplazaría un día la ventana de 28 días.
    hoy = pd.Timestamp(hoy).normalize() if hoy is not None else pd.Timestamp(config.hoy())
    inicio = hoy - pd.Timedelta(days=dias)                                         # Inicio de la ventana de análisis
    bloque = df[df["Fecha"] >= inicio]                                             # Actividades del mesociclo vigente

    carreras = bloque[                                                             # Carreras válidas dentro del bloque
        (bloque["Tipo de actividad"] == "Carrera")
        & (bloque["Distancia_km"] >= DISTANCIA_MINIMA_ESFUERZO)
        & (bloque["Ritmo (min/km)"].between(*RANGO_RITMO_VALIDO))
    ]
    if carreras.empty:                                                             # Sin carreras el modelo no puede evaluar
        return None                                                                # Señaliza la falta de datos recientes

    resistencia = bloque[bloque["Tipo de actividad"].isin(["Carrera", "Bicicleta"])]  # Base aeróbica del bloque

    return {                                                                       # Métricas homologadas al dataset de ajuste
        "ritmo_medio_mesociclo": carreras["Ritmo (min/km)"].mean(),                # Ritmo medio de las carreras recientes
        "km_semana_mesociclo": carreras["Distancia_km"].sum() / 4,                 # Volumen semanal de carrera
        "sesiones_mesociclo": len(carreras),                                       # Número de carreras del bloque
        "horas_semana_mesociclo": resistencia["Minutos"].sum() / 4 / 60,           # Horas aeróbicas semanales
    }

def ritmo_a_tiempo(ritmo_min_km, distancia_km, distancia_referencia=10.0):
    """
    Convierte un ritmo de referencia al tiempo total de una distancia objetivo
    aplicando la fórmula de Riegel, que corrige la degradación fisiológica del
    ritmo conforme aumenta la distancia (multiplicar linealmente sobreestima
    el rendimiento en pruebas largas).
    """
    if np.isnan(ritmo_min_km):                                                     # Sin ritmo no hay tiempo que calcular
        return np.nan                                                              # Devuelve un valor no disponible

    tiempo_referencia = ritmo_min_km * distancia_referencia                        # Tiempo en la distancia de referencia
    factor = (distancia_km / distancia_referencia) ** EXPONENTE_RIEGEL             # Penalización de Riegel por distancia
    return tiempo_referencia * factor                                              # Tiempo total estimado en minutos

def formatear_tiempo(minutos_decimales):
    """Transforma minutos decimales a formato de cronómetro (h:mm:ss)."""
    if minutos_decimales is None or np.isnan(minutos_decimales):                   # Protege contra valores no disponibles
        return "—"                                                                 # Devuelve un guion como marcador

    total_segundos = int(round(minutos_decimales * 60))                            # Convierte a segundos enteros
    horas, resto = divmod(total_segundos, 3600)                                    # Separa las horas completas
    minutos, segundos = divmod(resto, 60)                                          # Separa minutos y segundos

    if horas:                                                                      # Formato largo para pruebas de fondo
        return f"{horas}h {minutos:02d}m {segundos:02d}s"                          # Ejemplo: 1h 52m 30s
    return f"{minutos}m {segundos:02d}s"                                           # Ejemplo: 48m 15s

def fuera_de_rango_calibrado(distancia_km):
    """Indica si la distancia objetivo excede el rango con el que se entrenó el modelo."""
    return not (RANGO_CALIBRADO[0] <= distancia_km <= RANGO_CALIBRADO[1])          # True si es una extrapolación

def planificar_volumen(entrenamiento, metricas_actuales, ritmo_objetivo):
    """
    Planificador inverso: despeja de la ecuación de regresión el volumen semanal
    de carrera necesario para alcanzar el ritmo objetivo, manteniendo el resto de
    variables del bloque actual constantes.

    La regresión ajustada tiene la forma:
        ritmo = intercepto + b1*ritmo_medio + b2*km_semana + b3*horas

    Despejando km_semana para un ritmo objetivo dado se obtiene el volumen
    requerido. Devuelve un diccionario con el diagnóstico de la meta.
    """
    if entrenamiento is None or metricas_actuales is None:                         # Sin modelo o sin bloque vigente
        return {"viable": False, "mensaje": "No hay datos suficientes para planificar."}

    modelo = entrenamiento["modelo"]                                               # Recupera el modelo ya ajustado
    coeficientes = entrenamiento["coeficientes"]                                   # Pesos de cada variable independiente
    b_km = coeficientes["km_semana_mesociclo"]                                     # Coeficiente del volumen semanal

    if abs(b_km) < 1e-6 or b_km >= 0:                                              # El volumen debe reducir el ritmo
        return {                                                                   # Si el signo es contrario, no se despeja
            "viable": False,
            "mensaje": "El modelo no detecta una relación útil entre volumen y rendimiento en este historial.",
        }

    ritmo_actual = predecir_ritmo(entrenamiento, metricas_actuales)                # Ritmo estimado con la carga actual
    km_actual = metricas_actuales["km_semana_mesociclo"]                           # Volumen semanal vigente

    aporte_resto = (                                                               # Contribución de las variables fijas
        modelo.intercept_
        + coeficientes["ritmo_medio_mesociclo"] * metricas_actuales["ritmo_medio_mesociclo"]
        + coeficientes["horas_semana_mesociclo"] * metricas_actuales["horas_semana_mesociclo"]
    )
    km_necesarios = (ritmo_objetivo - aporte_resto) / b_km                         # Despeja el volumen semanal requerido

    if km_necesarios <= 0:                                                         # La meta se cumple sin volumen adicional
        return {                                                                   # Situación matemáticamente degenerada
            "viable": True, "km_necesarios": 0.0, "km_actual": km_actual,
            "incremento_pct": 0.0, "ritmo_actual": ritmo_actual, "riesgo": False,
            "mensaje": "Tu carga actual ya alcanza el objetivo según el modelo.",
        }

    incremento_pct = 100 * (km_necesarios - km_actual) / km_actual if km_actual > 0 else np.inf  # Salto de volumen exigido
    riesgo = incremento_pct > 10                                                   # Regla del 10 % semanal (prevención de lesiones)
    inalcanzable = km_necesarios > 4 * max(km_actual, 1)                           # Meta que exigiría cuadruplicar la carga

    if inalcanzable:                                                               # Objetivo fuera del alcance realista
        return {
            "viable": False,
            "mensaje": (f"El objetivo exigiría unos {km_necesarios:.1f} km/semana, más de cuatro veces "
                        f"tu volumen actual ({km_actual:.1f} km/semana). No es una meta alcanzable en un solo mesociclo."),
        }

    return {                                                                       # Plan viable con su diagnóstico de riesgo
        "viable": True,
        "km_necesarios": km_necesarios,                                            # Volumen semanal objetivo
        "km_actual": km_actual,                                                    # Volumen semanal vigente
        "incremento_pct": incremento_pct,                                          # Variación porcentual exigida
        "ritmo_actual": ritmo_actual,                                              # Ritmo estimado con la carga actual
        "riesgo": riesgo,                                                          # Alerta si supera el 10 % de incremento
        "mensaje": "Plan calculado correctamente.",                                # Estado de la operación
    }