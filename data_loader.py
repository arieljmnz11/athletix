"""
Módulo de Adquisición de Datos del SII Athletix.
Implementa la capa de entrada del sistema con una estrategia híbrida:
1. Carga inicial (backfill) desde el CSV histórico exportado de Strava.
2. Sincronización incremental vía API REST de Strava (OAuth 2.0), que
   descarga únicamente las actividades posteriores a la última registrada.
El resultado se persiste en Supabase (antes en un CSV local), actuando como
capa de almacenamiento que sobrevive a los reinicios del entorno en la nube.
"""
import os                                     # Acceso a variables de entorno
import pandas as pd                           # Manejo de datos tabulares
import re
import numpy as np
import requests                               # Cliente HTTP para la API de Strava
import metricas_fc
from dotenv import load_dotenv                # Carga del archivo .env local
from supabase import create_client, Client    # Cliente de la base de datos en la nube

load_dotenv()  # Lee el .env local; en la nube no hace nada (las llaves ya están en Secrets)

RUTA_CSV = "actividades_strava.csv"  # CSV histórico exportado de Strava (se queda en el repo)
MARGEN_SINCRONIZACION = 3            # Días de solape al descargar novedades
URL_TOKEN = "https://www.strava.com/oauth/token"
URL_ACTIVIDADES = "https://www.strava.com/api/v3/athlete/activities"
TABLA_FC = "fc_manual"
TAMANO_PAGINA = 1000
TAMANO_LOTE_UPSERT = 500

TABLA_SYNC = "actividades_sincronizadas"  # Tabla de Supabase que reemplaza al CSV incremental

# Traduce entre el nombre de columna que usa el DataFrame interno y el de la tabla en Supabase
MAPA_COLUMNAS_SYNC = {
    "Fecha de la actividad": "fecha_actividad",
    "Tipo de actividad": "tipo_actividad",
    "Distancia.1": "distancia_m",
    "Tiempo en movimiento": "tiempo_movimiento_s",
    "Tiempo transcurrido": "tiempo_transcurrido_s",
    "Desnivel positivo": "desnivel_positivo_m",
    "Ritmo cardiaco promedio": "fc_promedio",
    "Ritmo cardiaco máximo": "fc_maxima",
    "Velocidad promedio": "velocidad_promedio",
}
# Traducción de los tipos de deporte que devuelve la API (inglés) al formato del CSV (español)
TIPOS_API_A_CSV = {
    "Run": "Carrera", "TrailRun": "Carrera", "VirtualRun": "Carrera",
    "Ride": "Bicicleta", "VirtualRide": "Bicicleta", "MountainBikeRide": "Bicicleta",
    "GravelRide": "Bicicleta", "EBikeRide": "Bicicleta",
    "Walk": "Caminata", "Hike": "Senderismo", "Swim": "Natación",
    "Workout": "Entrenamiento", "WeightTraining": "Entrenamiento con pesas",
    "Rowing": "Remo",
}

_supabase: Client | None = None  # Cliente cacheado; se crea una sola vez por sesión

def _cliente_supabase():
    """Crea (o reutiliza) el cliente de Supabase a partir de las credenciales del entorno."""
    global _supabase
    if _supabase is None:
        _supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    return _supabase

def credenciales_strava_disponibles():
    """Indica si las tres credenciales de Strava están presentes en el .env."""
    claves = ["STRAVA_CLIENT_ID", "STRAVA_CLIENT_SECRET", "STRAVA_REFRESH_TOKEN"]
    return all(os.getenv(k) for k in claves)

def _obtener_access_token():
    """Canjea el refresh token por un access token temporal (válido ~6 horas)."""
    payload = {
        "client_id": os.getenv("STRAVA_CLIENT_ID"),
        "client_secret": os.getenv("STRAVA_CLIENT_SECRET"),
        "refresh_token": os.getenv("STRAVA_REFRESH_TOKEN"),
        "grant_type": "refresh_token",
    }
    respuesta = requests.post(URL_TOKEN, data=payload, timeout=30)
    respuesta.raise_for_status()
    return respuesta.json().get("access_token")

def _descargar_actividades(access_token, desde_epoch=None, max_paginas=15):
    """
    Descarga actividades paginando la API (200 por página, tope de rate limit).
    Si se indica 'desde_epoch', solo trae actividades posteriores a esa fecha.
    """
    cabecera = {"Authorization": f"Bearer {access_token}"}
    acumulado = []
    for pagina in range(1, max_paginas + 1):
        parametros = {"per_page": 200, "page": pagina}
        if desde_epoch:
            parametros["after"] = int(desde_epoch)
        respuesta = requests.get(URL_ACTIVIDADES, headers=cabecera, params=parametros, timeout=30)
        respuesta.raise_for_status()
        lote = respuesta.json()
        if not lote:
            break
        acumulado.extend(lote)
    return acumulado

def _json_a_formato_csv(actividades):
    """Convierte el JSON de la API al mismo esquema de columnas que usaba el CSV."""
    if not actividades:
        return pd.DataFrame()
    df = pd.json_normalize(actividades)
    tipo_origen = "sport_type" if "sport_type" in df.columns else "type"
    salida = pd.DataFrame()
    salida["Fecha de la actividad"] = df.get("start_date_local")
    salida["Tipo de actividad"] = df[tipo_origen].map(TIPOS_API_A_CSV).fillna(df[tipo_origen])
    salida["Distancia.1"] = df.get("distance")
    salida["Tiempo en movimiento"] = df.get("moving_time")
    # El tiempo transcurrido incluye las paradas; es el tiempo oficial de una competición,
    # mientras que el de movimiento es el que refleja la carga real de entrenamiento.
    salida["Tiempo transcurrido"] = df.get("elapsed_time")
    salida["Desnivel positivo"] = df.get("total_elevation_gain")
    salida["Ritmo cardiaco promedio"] = df.get("average_heartrate")
    # El máximo por actividad es un suelo observado de la FC máxima real del atleta,
    # más informado que una estimación por edad.
    salida["Ritmo cardiaco máximo"] = df.get("max_heartrate")
    salida["Velocidad promedio"] = df.get("average_speed")
    return salida

def _df_a_registros_sync(df):
    """Traduce el DataFrame homologado a una lista de diccionarios lista para Supabase."""
    tabla = df.rename(columns=MAPA_COLUMNAS_SYNC)[list(MAPA_COLUMNAS_SYNC.values())]
    registros = tabla.to_dict(orient="records")
    # to_dict() conserva los NaN como float('nan'); hay que limpiarlos aquí, ya como
    # objetos Python sueltos, porque dentro de una columna float64 el .where(..., None)
    # no funciona: pandas no puede guardar None en una columna numérica y lo revierte a NaN.
    return [
        {clave: (None if isinstance(valor, float) and pd.isna(valor) else valor)
         for clave, valor in registro.items()}
        for registro in registros
    ]

def _leer_tabla_paginada(tabla, columnas, columna_filtro=None, valor_filtro=None):
    """Descarga una tabla completa sorteando el tope de 1000 filas por petición de la API.

    Args:
        tabla (str): Nombre de la tabla en Supabase.
        columnas (str): Lista de columnas en formato de la API.
        columna_filtro (str | None): Columna sobre la que filtrar, si aplica.
        valor_filtro: Valor exacto que debe tomar esa columna.
    Returns:
        list[dict]: Todas las filas encontradas.
    """
    filas = []
    desde = 0
    while True:
        consulta = _cliente_supabase().table(tabla).select(columnas)
        if columna_filtro is not None:
            consulta = consulta.eq(columna_filtro, valor_filtro)
        lote = consulta.order("id").range(desde, desde + TAMANO_PAGINA - 1).execute().data
        if not lote:
            break
        filas.extend(lote)
        if len(lote) < TAMANO_PAGINA:
            break
        desde += TAMANO_PAGINA
    return filas

def _leer_sync_supabase():
    """Descarga lo sincronizado hasta ahora y lo devuelve con los nombres de columna originales."""
    filas = _leer_tabla_paginada(TABLA_SYNC, "*")
    if not filas:
        return pd.DataFrame()
    tabla = pd.DataFrame(filas)
    mapa_inverso = {v: k for k, v in MAPA_COLUMNAS_SYNC.items()}
    tabla = tabla.rename(columns=mapa_inverso)
    return tabla[[c for c in MAPA_COLUMNAS_SYNC.keys() if c in tabla.columns]]

def _hhmmss_a_segundos(valor):
    """Convierte 'HH:MM:SS' o 'MM:SS' a segundos, o None si el texto no es un tiempo."""
    partes = valor.split(":")
    if not 2 <= len(partes) <= 3:
        return None
    try:
        numeros = [int(p) for p in partes]
    except ValueError:
        return None
    while len(numeros) < 3:
        numeros.insert(0, 0)
    return numeros[0] * 3600 + numeros[1] * 60 + numeros[2]

def parsear_serie_fc(texto):
    """Convierte el texto copiado del reloj en una serie ordenada de tiempo y pulso.

    Args:
        texto (str): Filas con tiempo y pulso separados por tabulador, coma o espacios.
    Returns:
        pd.DataFrame: Columnas 'tiempo_s' y 'fc_ppm', ordenadas y sin tiempos repetidos.
    """
    filas = []
    for linea in texto.splitlines():
        # Las líneas de cabecera y los separadores se descartan solos al fallar el parseo.
        partes = [p for p in re.split(r"[\t,;]|\s+", linea.strip()) if p]
        if len(partes) < 2:
            continue
        segundos = _hhmmss_a_segundos(partes[0])
        if segundos is None:
            continue
        try:
            pulso = int(round(float(partes[1].replace(",", "."))))
        except ValueError:
            continue
        filas.append({"tiempo_s": segundos, "fc_ppm": pulso})
    if not filas:
        return pd.DataFrame(columns=["tiempo_s", "fc_ppm"])
    serie = pd.DataFrame(filas).drop_duplicates(subset="tiempo_s", keep="last")
    return serie.sort_values("tiempo_s").reset_index(drop=True)

def guardar_fc_manual(fecha_actividad, serie):
    """Persiste la serie de pulso de una actividad reemplazando las muestras repetidas.

    Args:
        fecha_actividad (str): Clave de la actividad, idéntica a la que guarda Strava.
        serie (pd.DataFrame): Salida de parsear_serie_fc.
    Returns:
        tuple[int, str]: Muestras guardadas y mensaje de estado.
    """
    if serie.empty:
        return 0, "No se reconoció ninguna muestra en el texto pegado."
    registros = [
        {"fecha_actividad": fecha_actividad, "tiempo_s": int(f.tiempo_s), "fc_ppm": int(f.fc_ppm)}
        for f in serie.itertuples()
    ]
    try:
        _cliente_supabase().table(TABLA_FC).upsert(
            registros, on_conflict="fecha_actividad,tiempo_s"
        ).execute()
        return len(registros), f"Se guardaron {len(registros)} muestras de pulso."
    except Exception as error:
        return 0, f"No se pudo guardar la serie: {error}"

def leer_fc_manual(fecha_actividad=None):
    """Recupera las muestras de pulso cargadas a mano, de una actividad o de todas.

    Args:
        fecha_actividad (str | None): Clave de la actividad, o None para leerlas todas.
    Returns:
        pd.DataFrame: Columnas 'fecha_actividad', 'tiempo_s' y 'fc_ppm', ordenadas
            cronológicamente dentro de cada actividad.
    """
    columnas_vacias = ["fecha_actividad", "tiempo_s", "fc_ppm"]
    try:
        filtro = "fecha_actividad" if fecha_actividad is not None else None
        filas = _leer_tabla_paginada(TABLA_FC, "id, fecha_actividad, tiempo_s, fc_ppm",
                                     filtro, fecha_actividad)
    except Exception:
        return pd.DataFrame(columns=columnas_vacias)
    if not filas:
        return pd.DataFrame(columns=columnas_vacias)

    # La lectura paginada ordena por id, que es el orden de inserción: al reescribir una
    # serie las muestras nuevas quedan al final y np.diff daría duraciones negativas.
    tabla = pd.DataFrame(filas)[columnas_vacias]
    return tabla.sort_values(["fecha_actividad", "tiempo_s"]).reset_index(drop=True)

def borrar_fc_manual(fecha_actividad):
    """Elimina todas las muestras de pulso cargadas para una actividad.

    Args:
        fecha_actividad (str): Clave de la actividad cuya serie se descarta.
    Returns:
        tuple[int, str]: Muestras eliminadas y mensaje de estado.
    """
    try:
        respuesta = _cliente_supabase().table(TABLA_FC).delete().eq(
            "fecha_actividad", fecha_actividad
        ).execute()
        borradas = len(respuesta.data or [])
        return borradas, f"Se eliminaron {borradas} muestras de {fecha_actividad}."
    except Exception as error:
        return 0, f"No se pudo borrar la serie: {error}"

def resumen_fc_manual():
    """Devuelve FC media ponderada y FC máxima por actividad con serie cargada."""
    columnas_vacias = ["fecha_actividad", "fc_media_manual", "fc_maxima_manual"]
    muestras = leer_fc_manual()
    if muestras.empty:
        return pd.DataFrame(columns=columnas_vacias)
    filas = []
    for clave, grupo in muestras.groupby("fecha_actividad"):
        grupo = grupo.sort_values("tiempo_s")
        tiempos = grupo["tiempo_s"].to_numpy(dtype=float)
        pulsos = grupo["fc_ppm"].to_numpy(dtype=float)
        filas.append({
            "fecha_actividad": clave,
            "fc_media_manual": metricas_fc.media_ponderada_por_tiempo(tiempos, pulsos),
            "fc_maxima_manual": float(pulsos.max()),
        })
    return pd.DataFrame(filas)

def listar_actividades_sin_fc():
    """Lista las actividades sincronizadas sin pulso, para saber dónde cargar la serie."""
    datos = _leer_sync_supabase()
    if datos.empty or "Ritmo cardiaco promedio" not in datos.columns:
        return pd.DataFrame(columns=["Fecha de la actividad", "Tipo de actividad"])
    sin_fc = datos[datos["Ritmo cardiaco promedio"].isna()]
    return sin_fc[["Fecha de la actividad", "Tipo de actividad"]].sort_values(
        "Fecha de la actividad", ascending=False
    )

def sincronizar_con_strava(fecha_ultima_actividad=None):
    """
    Descarga de la API únicamente las actividades posteriores a la última que ya
    se tiene registrada y las guarda en Supabase mediante upsert. Como
    'fecha_actividad' es columna única, un registro repetido se sobrescribe en
    vez de duplicarse — el mismo efecto que antes lograba drop_duplicates().
    Llamarla sin argumento descarga el histórico completo, útil para rellenar
    columnas añadidas al esquema después de las primeras sincronizaciones.
    Devuelve una tupla (numero_de_actividades_nuevas, mensaje_de_estado).
    """
    if not credenciales_strava_disponibles():
        return 0, "Faltan credenciales de Strava en el archivo .env."
    try:
        token = _obtener_access_token()
        desde = None
        if fecha_ultima_actividad is not None and pd.notna(fecha_ultima_actividad):
            # Se retrocede un margen de seguridad porque el parámetro 'after' de la API filtra
            # sobre la hora UTC, mientras que las fechas almacenadas son hora local. El upsert
            # por 'fecha_actividad' evita que este solape genere duplicados.
            corte = pd.Timestamp(fecha_ultima_actividad) - pd.Timedelta(days=MARGEN_SINCRONIZACION)
            desde = corte.timestamp()
        crudas = _descargar_actividades(token, desde_epoch=desde)
        nuevas = _json_a_formato_csv(crudas)
        if nuevas.empty:
            return 0, "Sin actividades nuevas. Ya estás al día."

        # Strava admite dos actividades con la misma hora de inicio, pero esa columna es
        # la clave única de la tabla y Postgres rechaza un lote que la repita. Se conserva
        # la última, igual que hace después el filtrado por fecha del backend.
        nuevas = nuevas.drop_duplicates(subset="Fecha de la actividad", keep="last")

        registros = _df_a_registros_sync(nuevas)
        # El histórico completo supera el tamaño cómodo de una sola petición, así que
        # el upsert se envía por lotes.
        for inicio in range(0, len(registros), TAMANO_LOTE_UPSERT):
            _cliente_supabase().table(TABLA_SYNC).upsert(
                registros[inicio:inicio + TAMANO_LOTE_UPSERT], on_conflict="fecha_actividad"
            ).execute()
        return len(registros), f"Se sincronizaron {len(registros)} actividades."
    except requests.exceptions.HTTPError as error:
        return 0, f"Strava rechazó la petición ({error.response.status_code}). Revisa tus credenciales."
    except Exception as error:
        return 0, f"No se pudo sincronizar: {error}"

def leer_fuentes_crudas():
    """
    Une el CSV histórico (local, en el repo) con las actividades sincronizadas
    (Supabase). Devuelve un único DataFrame crudo, o None si no hay ninguna fuente.
    """
    # El export histórico trae 103 columnas y la sincronización solo 8. Recortando
    # ambas fuentes al mismo esquema se evita generar columnas vacías al unirlas.
    # El tiempo transcurrido aparece duplicado en el export igual que la distancia,
    # por eso se admiten las dos variantes y el backend elige la numérica.
    columnas_utiles = [
        "Fecha de la actividad", "Tipo de actividad", "Distancia", "Distancia.1",
        "Tiempo en movimiento", "Tiempo transcurrido", "Tiempo transcurrido.1",
        "Desnivel positivo", "Ritmo cardiaco promedio", "Velocidad promedio",
    ]
    fuentes = []

    if os.path.exists(RUTA_CSV):
        datos = pd.read_csv(RUTA_CSV, low_memory=False)
        presentes = [c for c in columnas_utiles if c in datos.columns]
        datos = datos[presentes].dropna(axis=1, how="all")
        if not datos.empty:
            # La procedencia se conserva porque el solape entre fuentes no se puede
            # detectar comparando fechas: cada una las guarda en un huso distinto.
            fuentes.append(datos.assign(Origen="csv"))

    datos_sync = _leer_sync_supabase()
    if not datos_sync.empty:
        presentes = [c for c in columnas_utiles if c in datos_sync.columns]
        datos_sync = datos_sync[presentes].dropna(axis=1, how="all")
        if not datos_sync.empty:
            fuentes.append(datos_sync.assign(Origen="api"))

    if not fuentes:
        return None
    unido = pd.concat(fuentes, ignore_index=True)
    for columna in columnas_utiles:
        if columna not in unido.columns:
            unido[columna] = pd.NA

    # La FC manual solo rellena huecos: si Strava trajo pulso propio, ese dato manda.
    resumen = resumen_fc_manual()
    if not resumen.empty:
        mapa_media = dict(zip(resumen["fecha_actividad"], resumen["fc_media_manual"]))
        aporte = unido["Fecha de la actividad"].astype(str).map(mapa_media)
        unido["Ritmo cardiaco promedio"] = unido["Ritmo cardiaco promedio"].fillna(aporte)

    return unido