"""
Módulo Backend del SII Athletix.

Procesa los datos crudos de Strava: normaliza columnas duplicadas, descarta
registros inválidos, calcula la carga de entrenamiento sin falsear los datos
faltantes y deriva el indicador ACWR sobre una serie diaria real.

No depende de Streamlit, por lo que puede probarse de forma independiente.
"""
import numpy as np                                                                 # Operaciones numéricas vectorizadas
import pandas as pd                                                                # Manejo y análisis de datos tabulares
import config                                                                      # Zona horaria y utilidades de fecha
import data_loader                                                                 # Capa de adquisición de datos

# Variables globales de referencia para la limpieza y el análisis de datos
DEPORTES_DISTANCIA = ["Carrera", "Bicicleta", "Caminata", "Senderismo"]            # Deportes con distancia GPS fiable

# Deportes estáticos: el GPS no aporta distancia y la métrica interpretable es la duración
SIN_DISTANCIA = ["Entrenamiento", "Entrenamiento con pesas"]

RANGOS_RITMO = {                                                                   # Rangos plausibles de ritmo (min/km)
    "Carrera": (2.5, 12.0),                                                        # Descarta paradas y errores de reloj
    "Caminata": (7.0, 25.0),                                                       # Caminata y senderismo son más lentos
    "Senderismo": (7.0, 30.0),                                                     # Terreno de montaña, ritmos altos
    "Bicicleta": (1.0, 10.0),                                                      # Ciclismo es mucho más rápido por km
}

MESES_ES_EN = {                                                                    # Traducción de meses español -> inglés
    "ene": "jan", "abr": "apr", "ago": "aug", "dic": "dec",                        # Solo difieren estas cuatro abreviaturas
}

DIAS_MESOCICLO = 28                                                                # Bloque de carga de 4 semanas

# Umbrales de referencia del ratio agudo:crónico (Gabbett, 2016)
UMBRAL_OPTIMO_BAJO = 0.8                                                           # Por debajo hay desentrenamiento
UMBRAL_OPTIMO_ALTO = 1.3                                                           # Límite superior de la zona óptima
UMBRAL_RIESGO = 1.5                                                                # Por encima, alto riesgo de lesión

def _fecha_referencia(hoy=None):
    """
    Normaliza la fecha desde la que se miden las ventanas móviles.

    Cuando no se indica una fecha se toma el día del atleta y no el del servidor:
    en la nube el proceso corre en UTC, que de noche ya va un día por delante de
    Ecuador y desplazaría las ventanas de 7 y 28 días.
    """
    if hoy is not None:                                                            # Permite fijar la fecha en pruebas
        return pd.Timestamp(hoy).normalize()                                       # Respeta la fecha recibida
    return pd.Timestamp(config.hoy())                                              # Medianoche del día local del atleta

# Limpieza y normalización de datos
def _normalizar_distancia(df):
    """
    Resuelve el conflicto de columnas duplicadas del export de Strava.
    El CSV trae 'Distancia' dos veces: la primera en km como texto con coma
    decimal y la segunda (renombrada a 'Distancia.1' por pandas) en metros.
    Se prioriza la versión en metros por ser numérica y fiable.
    """
    if "Distancia.1" in df.columns:                                                # Versión en metros (float limpio)
        metros = pd.to_numeric(df["Distancia.1"], errors="coerce")                 # Fuerza el tipo numérico
        return metros / 1000                                                       # Convierte metros a kilómetros

    texto = df["Distancia"].astype(str).str.replace(",", ".", regex=False)         # Versión en km con coma decimal
    return pd.to_numeric(texto, errors="coerce")                                   # Devuelve los kilómetros ya numéricos

def _normalizar_tiempo_transcurrido(df):
    """
    Devuelve el tiempo transcurrido en segundos combinando las variantes posibles.

    El export de Strava duplica esta columna igual que la distancia, y las
    actividades sincronizadas antes de añadir el campo al esquema no lo traen.
    Se recorren las candidatas y se conserva el primer valor numérico de cada fila.
    """
    resultado = pd.Series(np.nan, index=df.index, dtype="float64")                 # Contenedor del resultado

    for columna in ("Tiempo transcurrido", "Tiempo transcurrido.1"):               # Variantes que puede traer el origen
        if columna in df.columns:                                                  # Solo si la fuente la incluye
            candidata = pd.to_numeric(df[columna], errors="coerce")                # Fuerza el tipo numérico
            resultado = resultado.fillna(candidata)                                # Completa los huecos pendientes

    return resultado                                                               # Puede contener NaN si no hay dato

def _parsear_fechas(serie):
    """
    Convierte las fechas de ambas fuentes a datetime sin zona horaria.

    Las dos fuentes usan formatos incompatibles y deben parsearse por separado:
        - CSV exportado: '27 jun 2026, 13:17:16' (día primero, sin zona horaria).
        - API de Strava: '2026-07-12T06:40:01Z' (formato ISO con zona horaria).

    Aplicar 'dayfirst=True' a una fecha ISO invertiría el día y el mes, y mezclar
    fechas con y sin zona horaria impide ordenarlas en una misma serie.
    """
    texto = serie.astype(str).str.lower()                                          # Homogeneiza a minúsculas
    for esp, eng in MESES_ES_EN.items():                                           # Recorre las abreviaturas divergentes
        texto = texto.str.replace(esp, eng, regex=False)                           # Traduce el mes al formato inglés

    es_iso = texto.str.match(r"^\d{4}-\d{2}-\d{2}")                                # Detecta el formato ISO de la API
    fechas = pd.Series(pd.NaT, index=texto.index, dtype="datetime64[ns]")          # Contenedor unificado del resultado

    if es_iso.any():                                                               # Solo si hay registros de la API
        iso = pd.to_datetime(texto[es_iso], errors="coerce", utc=True)             # Parsea ISO respetando la zona horaria
        fechas.loc[es_iso] = iso.dt.tz_localize(None)                              # Descarta la zona para poder comparar

    if (~es_iso).any():                                                            # Solo si hay registros del CSV
        csv = pd.to_datetime(texto[~es_iso], errors="coerce",                      # Parsea el formato español del export
                             dayfirst=True, format="mixed")                        # El día precede al mes en el CSV
        fechas.loc[~es_iso] = csv                                                  # Integra el resultado en la serie

    return fechas                                                                  # Devuelve todas las fechas homogeneizadas

def _filtrar_ritmos_absurdos(df):
    """
    Anula los ritmos fisiológicamente imposibles. Son consecuencia de deportes
    sin distancia (pesas, natación en piscina) y de relojes que se olvidaron
    detener, y son la causa de los picos que distorsionaban las gráficas.
    """
    valido = pd.Series(False, index=df.index)                                      # Máscara inicial: nada es válido

    for deporte, (minimo, maximo) in RANGOS_RITMO.items():                         # Aplica el rango propio de cada deporte
        es_deporte = df["Tipo de actividad"] == deporte                            # Selecciona las filas del deporte
        en_rango = df["Ritmo (min/km)"].between(minimo, maximo)                    # Comprueba el rango plausible
        valido |= (es_deporte & en_rango)                                          # Marca como válidas las que cumplen

    df.loc[~valido, "Ritmo (min/km)"] = np.nan                                     # Anula el resto en lugar de borrar filas
    return df                                                                      # Devuelve el DataFrame corregido

def _calcular_carga(df):
    """
    Calcula la carga de entrenamiento SIN imputar la frecuencia cardíaca.

    El 60 % de las actividades del histórico no tiene FC registrada, por lo que
    rellenarla con la media global falsearía la mayor parte del dataset. En su
    lugar se usa el TRIMP real donde hay pulsómetro y, donde no lo hay, se
    estima la carga a partir de la duración escalada por el factor mediano
    (carga por minuto) observado en ese mismo deporte.

    La duración empleada es siempre la de movimiento: el tiempo detenido no
    genera fatiga, aunque sí cuente para el resultado oficial de una competición.
    """
    df["Minutos"] = df["Tiempo en movimiento"] / 60                                # Convierte la duración a minutos
    df["Tiene FC"] = df["Ritmo cardiaco promedio"].notna()                         # Marca la cobertura real del pulsómetro

    df["Carga"] = np.nan                                                           # Inicializa la columna de carga
    con_fc = df["Tiene FC"]                                                        # Selecciona las filas con pulsómetro
    df.loc[con_fc, "Carga"] = (                                                     # TRIMP simplificado (duración x intensidad)
        df.loc[con_fc, "Minutos"] * df.loc[con_fc, "Ritmo cardiaco promedio"] / 100
    )

    for deporte in df["Tipo de actividad"].dropna().unique():                      # Calibra un factor por cada deporte
        es_deporte = df["Tipo de actividad"] == deporte                            # Filas del deporte en cuestión
        referencia = df[es_deporte & con_fc]                                       # Sesiones con FC de ese deporte

        if len(referencia) >= 5 and referencia["Minutos"].sum() > 0:               # Requiere muestra mínima para calibrar
            factor = (referencia["Carga"] / referencia["Minutos"]).median()        # Carga mediana por minuto del deporte
        else:
            factor = 1.4                                                           # Respaldo equivalente a unas 140 ppm

        faltan = es_deporte & ~con_fc                                              # Sesiones sin FC de ese deporte
        df.loc[faltan, "Carga"] = df.loc[faltan, "Minutos"] * factor               # Estima la carga desde la duración

    return df                                                                      # Devuelve el DataFrame con la carga

def cargar_y_procesar_datos():
    """
    Orquesta la carga completa: adquiere las fuentes, normaliza, limpia, calcula
    variables derivadas y ordena temporalmente.

    Devuelve un DataFrame listo para analizar, o None si no hay datos.
    """
    crudo = data_loader.leer_fuentes_crudas()                                      # Une CSV histórico y sincronización API
    if crudo is None or crudo.empty:                                               # Si no existe ninguna fuente de datos
        return None                                                                # Señaliza la ausencia al frontend

    df = pd.DataFrame()                                                            # Contenedor del dataset limpio
    df["Fecha"] = _parsear_fechas(crudo["Fecha de la actividad"])                  # Convierte las fechas a datetime
    df["Tipo de actividad"] = crudo["Tipo de actividad"]                           # Conserva el deporte practicado
    df["Distancia_km"] = _normalizar_distancia(crudo)                              # Resuelve las columnas duplicadas
    df["Tiempo en movimiento"] = pd.to_numeric(crudo["Tiempo en movimiento"], errors="coerce")  # Duración en segundos
    df["Tiempo transcurrido"] = _normalizar_tiempo_transcurrido(crudo)             # Duración con paradas incluidas
    df["Ritmo cardiaco promedio"] = pd.to_numeric(crudo["Ritmo cardiaco promedio"], errors="coerce")  # FC media real
    df["Desnivel positivo"] = pd.to_numeric(crudo["Desnivel positivo"], errors="coerce").fillna(0)  # Desnivel acumulado
    df["Origen"] = crudo["Origen"] if "Origen" in crudo.columns else "csv"

    df = df.dropna(subset=["Fecha", "Tiempo en movimiento"])                       # Descarta registros sin fecha ni duración
    df = df[df["Tiempo en movimiento"] > 0]                                        # Elimina actividades de duración nula

    # El export de Strava fecha en UTC y la API en hora local, así que una misma
    # actividad no coincide entre ambas y se contaría dos veces. Como la sincronización
    # ya cubre todo el histórico, la API manda y el CSV solo aporta lo que la preceda.
    if (df["Origen"] == "api").any():
        inicio_api = df.loc[df["Origen"] == "api", "Fecha"].min()
        df = df[(df["Origen"] == "api") | (df["Fecha"] < inicio_api)]
        
    # Ante una actividad presente en ambas fuentes se conserva la de la API, que se
    # concatena después y es la que lleva enlazada la frecuencia cardíaca manual.
    df = df.drop_duplicates(subset=["Fecha"], keep="last")
    df = df.sort_values("Fecha").reset_index(drop=True)                            # Ordena cronológicamente

    # Las actividades anteriores a la incorporación del campo no traen tiempo
    # transcurrido; se asume igual al de movimiento para no propagar huecos.
    df["Tiempo transcurrido"] = df["Tiempo transcurrido"].fillna(df["Tiempo en movimiento"])

    df["Minutos"] = df["Tiempo en movimiento"] / 60                                # Duración en minutos
    df["Minutos transcurridos"] = df["Tiempo transcurrido"] / 60                   # Duración total en minutos
    ritmo = df["Minutos"] / df["Distancia_km"].replace(0, np.nan)                  # Ritmo bruto evitando dividir por cero
    df["Ritmo (min/km)"] = ritmo.replace([np.inf, -np.inf], np.nan)                # Elimina infinitos residuales
    df = _filtrar_ritmos_absurdos(df)                                              # Anula los ritmos fuera de rango
    df["Velocidad (km/h)"] = df["Distancia_km"] / (df["Minutos"] / 60)             # Velocidad media en km/h
    df.loc[df["Ritmo (min/km)"].isna(), "Velocidad (km/h)"] = np.nan               # Coherencia con el filtro anterior

    df = _calcular_carga(df)                                                       # Calcula la carga sin imputar la FC

    df["Semana"] = df["Fecha"].dt.to_period("W").dt.start_time                     # Etiqueta la semana calendario (lunes)
    df["Año"] = df["Fecha"].dt.year                                                # Año, útil para los filtros
    dias_desde_inicio = (df["Fecha"] - df["Fecha"].min()).dt.days                  # Días transcurridos desde el origen
    df["Mesociclo"] = dias_desde_inicio // DIAS_MESOCICLO                          # Bloque de carga de 4 semanas

    return df                                                                      # Devuelve el dataset procesado

# Funciones de análisis y agregación
def serie_carga_diaria(df, fecha_corte=None):
    """
    Convierte las actividades en una serie diaria continua de carga, rellenando
    con cero los días de descanso. Este paso es imprescindible: el ACWR se
    define sobre ventanas de DÍAS, no sobre las últimas N actividades.
    """
    if df.empty:                                                                   # Sin actividades no hay serie posible
        return pd.DataFrame()                                                      # Devuelve un DataFrame vacío

    fin = fecha_corte if fecha_corte is not None else df["Fecha"].max()            # Permite calcular "a día de hoy"
    diaria = df.set_index("Fecha")["Carga"].resample("D").sum()                    # Suma la carga de cada día natural
    calendario = pd.date_range(df["Fecha"].min().normalize(), pd.Timestamp(fin).normalize(), freq="D")  # Calendario completo
    diaria = diaria.reindex(calendario, fill_value=0)                              # Los días sin entrenar valen cero

    serie = pd.DataFrame({"Fecha": diaria.index, "Carga diaria": diaria.values})   # Estructura la serie resultante
    serie["Carga Aguda"] = serie["Carga diaria"].rolling(7, min_periods=1).mean()  # Fatiga: media de los últimos 7 días
    serie["Carga Cronica"] = serie["Carga diaria"].rolling(28, min_periods=1).mean()  # Condición: media de 28 días
    serie["ACWR"] = (serie["Carga Aguda"] / serie["Carga Cronica"]).replace([np.inf, -np.inf], np.nan)  # Ratio agudo:crónico

    return serie                                                                   # Devuelve la serie diaria completa

def estado_actual(df, hoy=None):
    """
    Calcula el estado de fatiga vigente a día de hoy sobre el histórico completo.

    Devuelve un diccionario con el ACWR, la etiqueta de riesgo y los días
    transcurridos desde la última actividad registrada.
    """
    hoy = _fecha_referencia(hoy)                                                   # Día local del atleta como referencia

    if df.empty:                                                                   # Sin datos no hay diagnóstico
        return {"acwr": np.nan, "estado": "Sin datos", "dias_inactivo": None}      # Devuelve un estado neutro

    ultima = df["Fecha"].max()                                                     # Fecha de la última actividad
    dias_inactivo = (hoy - ultima.normalize()).days                                # Días transcurridos sin entrenar

    serie = serie_carga_diaria(df, fecha_corte=hoy)                                # Serie diaria hasta la fecha actual
    if serie.empty or serie["Carga Cronica"].iloc[-1] == 0:                        # Sin carga crónica el ratio no aplica
        return {"acwr": np.nan, "estado": "Sin datos suficientes", "dias_inactivo": dias_inactivo}

    acwr = float(serie["ACWR"].iloc[-1])                                           # Último valor del ratio agudo:crónico

    if acwr > UMBRAL_RIESGO:                                                       # Sobrecarga aguda respecto a la base
        estado = "ALTO RIESGO DE LESIÓN"                                           # Zona roja de la literatura (Gabbett)
    elif acwr > UMBRAL_OPTIMO_ALTO:                                                # Franja intermedia de vigilancia
        estado = "PRECAUCIÓN (carga en ascenso)"                                   # Aviso previo a la zona de riesgo
    elif acwr < UMBRAL_OPTIMO_BAJO:                                                # Estímulo por debajo de lo necesario
        estado = "CARGA BAJA (desentrenamiento)"                                   # Pérdida progresiva de condición
    else:                                                                          # Franja recomendada 0.8 - 1.3
        estado = "ESTADO ÓPTIMO"                                                   # Zona segura de progresión

    return {"acwr": acwr, "estado": estado, "dias_inactivo": dias_inactivo}        # Devuelve el diagnóstico completo

def dias_sin_entrenar(df, hoy=None):
    """Días transcurridos desde la última actividad del subconjunto recibido."""
    if df.empty:                                                                   # Sin actividades no hay referencia
        return None                                                                # Señaliza la ausencia de datos

    hoy = _fecha_referencia(hoy)                                                   # Día local del atleta como referencia
    return (hoy - df["Fecha"].max().normalize()).days                              # Diferencia en días naturales

def formatear_ritmo(ritmo_min_km):
    """
    Convierte un ritmo decimal a formato de cronómetro (min:seg por kilómetro).

    Es necesario porque la parte decimal de un ritmo son centésimas de minuto y no
    segundos: 4.9 min/km no equivale a 4 min 90 s, sino a 4 min 54 s.
    """
    if pd.isna(ritmo_min_km):                                                      # Protege contra valores no disponibles
        return "—"                                                                 # Devuelve un guion como marcador

    total_segundos = int(round(ritmo_min_km * 60))                                 # Pasa el ritmo completo a segundos
    minutos, segundos = divmod(total_segundos, 60)                                 # Separa minutos enteros y segundos
    return f"{minutos}:{segundos:02d}"                                             # Devuelve el formato 4:54

def formatear_horas(horas_decimales):
    """Convierte un total de horas decimales a la forma '1466 h 30 min'."""
    if pd.isna(horas_decimales):                                                   # Protege contra valores no disponibles
        return "—"                                                                 # Devuelve un guion como marcador

    total_minutos = int(round(horas_decimales * 60))                               # Pasa el total a minutos enteros
    horas, minutos = divmod(total_minutos, 60)                                     # Separa horas completas y resto
    return f"{horas} h {minutos:02d} min"                                          # Devuelve el texto ya formateado

def formatear_duracion_corta(segundos):
    """
    Convierte una duración en segundos al formato compacto '2h43min' o '45min'.

    Se redondea al minuto más próximo porque los segundos no aportan información
    útil en las etiquetas y sí ocupan espacio en gráficos y tablas.
    """
    if segundos is None or pd.isna(segundos):                                      # Protege contra valores no disponibles
        return "—"                                                                 # Devuelve un guion como marcador

    minutos_totales = int(round(float(segundos) / 60))                             # Redondea al minuto más cercano
    horas, minutos = divmod(minutos_totales, 60)                                   # Separa horas completas y resto
    return f"{horas}h{minutos:02d}min" if horas else f"{minutos}min"               # Omite las horas cuando no las hay

def calcular_kpis(df, hoy=None):
    """Calcula los indicadores numéricos del panel principal."""
    hoy = _fecha_referencia(hoy)                                                   # Día local del atleta como referencia
    kpis = {"actividades": len(df)}                                                # Número total de actividades

    if df.empty:                                                                   # Protege contra filtros sin resultados
        return kpis                                                                # Devuelve solo el conteo

    ultimos_7 = df[df["Fecha"] >= hoy - pd.Timedelta(days=7)]                      # Ventana móvil de 7 días
    ultimos_28 = df[df["Fecha"] >= hoy - pd.Timedelta(days=28)]                    # Ventana móvil de 28 días

    kpis["km_7d"] = ultimos_7["Distancia_km"].sum()                                # Volumen de la última semana
    kpis["km_28d"] = ultimos_28["Distancia_km"].sum()                              # Volumen del último mesociclo
    kpis["km_total"] = df["Distancia_km"].sum()                                    # Volumen histórico acumulado
    kpis["desnivel_7d"] = ultimos_7["Desnivel positivo"].sum()                     # Desnivel de la última semana
    kpis["desnivel_28d"] = ultimos_28["Desnivel positivo"].sum()                   # Desnivel del último mesociclo
    kpis["horas_28d"] = ultimos_28["Minutos"].sum() / 60                           # Horas entrenadas en 28 días
    kpis["cobertura_fc"] = 100 * df["Tiene FC"].mean()                             # Porcentaje real de datos con pulsómetro

    ritmos = df["Ritmo (min/km)"].dropna()                                         # Ritmos válidos tras la limpieza
    kpis["mejor_ritmo"] = ritmos.min() if not ritmos.empty else np.nan             # Mejor ritmo del periodo analizado
    kpis["ritmo_medio"] = ritmos.mean() if not ritmos.empty else np.nan            # Ritmo medio del periodo analizado

    # Marcas separadas por deporte: el ciclismo domina el histórico y su ritmo por
    # kilómetro siempre sería el mejor, ocultando la marca real de carrera.
    solo_carrera = df[df["Tipo de actividad"] == "Carrera"]["Ritmo (min/km)"].dropna()  # Ritmos de carrera a pie
    kpis["mejor_ritmo_carrera"] = solo_carrera.min() if not solo_carrera.empty else np.nan  # Mejor marca corriendo

    solo_bici = df[df["Tipo de actividad"] == "Bicicleta"]["Velocidad (km/h)"].dropna()  # Velocidades en bicicleta
    kpis["mejor_velocidad_bici"] = solo_bici.max() if not solo_bici.empty else np.nan  # Mejor velocidad pedaleando

    return kpis                                                                    # Devuelve el diccionario de indicadores

def resumen_semanal(df):
    """Agrega volumen, carga, desnivel y duración por semana calendario y deporte."""
    if df.empty:                                                                   # Sin datos no hay agregación
        return pd.DataFrame()                                                      # Devuelve un DataFrame vacío
    semanal = df.groupby(["Semana", "Tipo de actividad"]).agg(                     # Agrupa por semana y deporte
        Kilometros=("Distancia_km", "sum"),                                        # Volumen semanal en kilómetros
        Carga=("Carga", "sum"),                                                    # Carga total acumulada en la semana
        Desnivel=("Desnivel positivo", "sum"),                                     # Desnivel positivo acumulado
        Minutos=("Minutos", "sum"),                                                # Duración en movimiento acumulada
    ).reset_index()                                                                # Devuelve el índice a columnas

    # Etiqueta legible del rango completo (lunes a domingo). Incluye el año porque el
    # gráfico la usa como categoría del eje y, sin él, las semanas homónimas de años
    # distintos se apilarían en una sola barra.
    fin_semana = semanal["Semana"] + pd.Timedelta(days=6)
    semanal["Rango"] = (semanal["Semana"].dt.strftime("%d/%m") + " - "
                        + fin_semana.dt.strftime("%d/%m") + " · "
                        + semanal["Semana"].dt.strftime("%Y"))

    # Solo se redondean las columnas numéricas: aplicar round() al DataFrame completo
    # incluiría la columna de fecha y pandas emitiría un aviso.
    numericas = ["Kilometros", "Carga", "Desnivel", "Minutos"]                     # Columnas que admiten redondeo
    semanal[numericas] = semanal[numericas].round(1)                               # Un decimal en todas ellas
    return semanal.sort_values("Semana")                                           # Ordena cronológicamente

def resumen_por_tipo(df):
    """Resume actividades, volumen y horas por tipo de deporte."""
    if df.empty:                                                                   # Sin datos no hay resumen
        return pd.DataFrame()                                                      # Devuelve un DataFrame vacío

    resumen = df.groupby("Tipo de actividad").agg(                                 # Agrupa por deporte practicado
        Actividades=("Fecha", "count"),                                            # Número de sesiones
        Kilometros=("Distancia_km", "sum"),                                        # Volumen total acumulado
        Desnivel=("Desnivel positivo", "sum"),                                     # Desnivel positivo acumulado
        Horas=("Minutos", lambda x: x.sum() / 60),                                 # Tiempo total en horas decimales
    ).reset_index().sort_values("Actividades", ascending=False)                    # Ordena de mayor a menor frecuencia

    resumen["Kilometros"] = resumen["Kilometros"].round(1)                          # Un decimal en el volumen
    resumen["Desnivel"] = resumen["Desnivel"].round(0)                              # El desnivel se expresa en metros
    resumen["Tiempo total"] = resumen["Horas"].apply(formatear_horas)               # Tiempo en formato horas y minutos
    resumen = resumen.drop(columns=["Horas"])                                       # Sustituye la versión decimal
    resumen = resumen.rename(columns={"Desnivel": "Desnivel (m)"})                  # Aclara la unidad en la cabecera

    return resumen.reset_index(drop=True)                                          # Reindexa el resultado final

def ultimas_actividades(df, n=10):
    """Devuelve las últimas n actividades con las columnas ya formateadas."""
    if df.empty:                                                                   # Sin datos no hay tabla
        return pd.DataFrame()                                                      # Devuelve un DataFrame vacío

    tabla = df.tail(n).sort_values("Fecha", ascending=False).copy()                # Toma las más recientes primero
    tabla["Fecha"] = tabla["Fecha"].dt.strftime("%d/%m/%Y")                        # Formatea la fecha de forma legible

    # El tiempo transcurrido se guarda ya formateado porque es el dato que interesa
    # leer de un vistazo; el de movimiento se conserva numérico para el agente.
    tabla["Tiempo total"] = tabla["Tiempo transcurrido"].apply(formatear_duracion_corta)

    # El ritmo (min/km) describe bien la carrera, pero en ciclismo la métrica
    # interpretable es la velocidad media, así que se muestra una u otra según el deporte.
    es_bici = tabla["Tipo de actividad"] == "Bicicleta"                            # Identifica las salidas en bicicleta
    tabla.loc[es_bici, "Ritmo (min/km)"] = np.nan                                  # Oculta el ritmo en ciclismo
    tabla.loc[~es_bici, "Velocidad (km/h)"] = np.nan                               # Oculta la velocidad en el resto

    columnas = ["Fecha", "Tipo de actividad", "Distancia_km", "Minutos",           # Columnas relevantes para el usuario
                "Tiempo total", "Desnivel positivo", "Ritmo (min/km)",
                "Velocidad (km/h)", "Carga"]
    tabla = tabla[columnas].round(1)                                               # Redondea para evitar decimales largos
    tabla["Ritmo (min/km)"] = tabla["Ritmo (min/km)"].apply(formatear_ritmo)       # Convierte el ritmo a min:seg
    return tabla.rename(columns={"Distancia_km": "Km", "Minutos": "Duración Minutos",           # Nombres cortos para la tabla
                                 "Desnivel positivo": "D+ (m)",
                                 "Ritmo (min/km)": "Ritmo (min:s/km)",
                                 "Velocidad (km/h)": "Velocidad (km/h)"})