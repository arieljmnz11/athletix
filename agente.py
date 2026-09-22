"""
Módulo del Agente LLM del SII Athletix.
Gestiona la capa conversacional del sistema:
- Persiste el historial de chat en Supabase, de modo que el contexto sobreviva
  al cierre de la aplicación y a los reinicios del entorno en la nube.
- Mantiene un diario de estado físico que se inyecta como hecho estructurado.
- Construye el contexto del sistema con los indicadores y la predicción
  calculados por los demás módulos, de forma que el agente interprete datos
  reales y no opere como un chatbot aislado.
"""
import os                                     # Acceso a variables de entorno
from datetime import datetime, timedelta
import config
import pandas as pd                           # Para la comprobación de valores nulos
from supabase import create_client, Client    # Cliente de la base de datos en la nube

MAX_MENSAJES = 20            # Ventana de historial enviada al modelo
MAX_ENTRADAS_DIARIO = 7      # Últimos días de estado que se inyectan al agente
RETENCION_DIARIO = 30        # Días de diario que se conservan en la base de datos
MODELO = "claude-sonnet-5"

_supabase: Client | None = None  # Cliente cacheado; se crea una sola vez por sesión

def _cliente_supabase():
    """Crea (o reutiliza) el cliente de Supabase a partir de las credenciales del entorno."""
    global _supabase
    if _supabase is None:
        _supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    return _supabase

# Persistencia del historial de chat en Supabase, solamente que se guarda el rol y el contenido de cada mensaje, no la fecha ni otros metadatos.

def cargar_historial():
    """Recupera desde Supabase el historial de chat de sesiones anteriores."""
    try:
        respuesta = _cliente_supabase().table("historial_chat").select("rol, contenido").order("id").execute()
        return [{"role": f["rol"], "content": f["contenido"]} for f in respuesta.data]
    except Exception:
        return []

def guardar_historial(mensajes):
    """Persiste el historial de chat en Supabase tras cada intercambio (sobrescribe todo, igual que antes)."""
    try:
        cliente = _cliente_supabase()
        cliente.table("historial_chat").delete().neq("id", 0).execute()  # Borra lo anterior (id siempre > 0)
        if mensajes:
            registros = [{"rol": m["role"], "contenido": m["content"]} for m in mensajes]
            cliente.table("historial_chat").insert(registros).execute()
    except Exception:
        pass

def borrar_historial():
    """Elimina la memoria conversacional almacenada."""
    try:
        _cliente_supabase().table("historial_chat").delete().neq("id", 0).execute()
    except Exception:
        pass

# Esta función gestiona el diario de estado físico que el deportista puede declarar, de modo que el agente pueda priorizarlo sobre los indicadores calculados automáticamente.

def cargar_diario():
    """Recupera desde Supabase el diario de estado físico registrado por el deportista."""
    try:
        respuesta = _cliente_supabase().table("diario_estado").select("fecha, estado, nota").order("fecha").execute()
        return [{"fecha": f["fecha"], "estado": f["estado"], "nota": f.get("nota") or ""} for f in respuesta.data]
    except Exception:
        return []

def registrar_estado(estado, nota="", retencion=None):
    """
    Añade o actualiza el estado físico del día en el diario. Este dato se inyecta
    después como hecho estructurado, en lugar de confiar en que el modelo lo
    deduzca del historial conversacional.
    """
    hoy = config.hoy().isoformat()

    try:
        cliente = _cliente_supabase()
        # 'fecha' es primary key: el upsert reemplaza el registro de hoy si ya existía
        cliente.table("diario_estado").upsert(
            {"fecha": hoy, "estado": estado, "nota": nota}, on_conflict="fecha"
        ).execute()
        # Se conserva un número de registros, no de días naturales: si se anota tres
        # veces por semana, treinta registros cubren unas diez semanas.
        respuesta = cliente.table("diario_estado").select("fecha").order("fecha", desc=True).execute()
        fechas_a_borrar = [f["fecha"] for f in respuesta.data[retencion or RETENCION_DIARIO:]]
        if fechas_a_borrar:
            cliente.table("diario_estado").delete().in_("fecha", fechas_a_borrar).execute()
    except Exception:
        pass
    return cargar_diario()

# Función que construye el contexto del sistema para el agente, combinando indicadores, predicciones y diario de estado físico.

def construir_contexto(kpis, diagnostico, prediccion, entrenamiento, diario,
                       actividades_recientes=None, cambios=None):
    """
    Ensambla el prompt de sistema con la salida real de los módulos analíticos.
    El agente recibe los indicadores, el resultado del componente predictivo, su
    fiabilidad declarada y el detalle de las últimas actividades individuales, de
    modo que pueda responder sobre una sesión concreta y no solo sobre agregados.
    """
    ahora = datetime.now(config.ZONA_HORARIA)
    # Cada día lleva su etiqueta relativa escrita: con una lista sin etiquetar el modelo
    # tenía que contar posiciones para encontrar "mañana" y a veces tomaba la tercera.
    etiquetas = ["Hoy", "Mañana", "Pasado mañana"] + [f"Dentro de {i} días" for i in range(3, 8)]
    calendario = [f"  - {etiqueta}: {config.fecha_en_texto(ahora.date() + timedelta(days=i))}"
                  for i, etiqueta in enumerate(etiquetas)]

    lineas = [
        "Eres un entrenador deportivo profesional que asesora a un atleta amateur.",
        "Recibes los indicadores calculados por un sistema de análisis de datos de Strava.",
        "Responde en español, de forma breve, concreta y sin introducciones largas.",
        "",
        # Sin hora a propósito: el modelo la usaba como excusa para decir que una carga
        # hecha "después" no le había llegado.
        "Todo lo que sigue se leyó de la base de datos al recibir el último mensaje del "
        "atleta. Si dice que acaba de cargar o corregir algo, ya está incluido aquí.",
        "",
        "CALENDARIO, ya resuelto. Úsalo tal cual y no recalcules ningún día:",
    ] + calendario + [
        "De cualquier otra fecha cita el día del mes, nunca el día de la semana.",
        "",
        "ESTADO ACTUAL DEL ATLETA (indicadores calculados por el sistema):",
    ]
    acwr = diagnostico.get("acwr")
    if acwr is not None and acwr == acwr:
        lineas.append(f"- Ratio de carga aguda:crónica (ACWR): {acwr:.2f} → {diagnostico['estado']}.")
        lineas.append(" Interpretación: por debajo de 0.8 hay desentrenamiento; por encima de 1.5, riesgo de lesión.")
    else:
        lineas.append("- ACWR: no disponible por falta de datos recientes.")
    dias = diagnostico.get("dias_inactivo")
    if dias is not None:
        lineas.append(f"- Días desde la última actividad registrada: {dias}.")
    lineas += [
        f"- Volumen últimos 7 días: {kpis.get('km_7d', 0):.1f} km.",
        f"- Volumen últimos 28 días: {kpis.get('km_28d', 0):.1f} km.",
        f"- Horas entrenadas en 28 días: {kpis.get('horas_28d', 0):.1f} h.",
        f"- Cobertura de pulsómetro en el histórico: {kpis.get('cobertura_fc', 0):.0f} %.",
    ]
    if actividades_recientes is not None and not actividades_recientes.empty:
        lineas.append("")
        lineas.append(f"ÚLTIMAS {len(actividades_recientes)} ACTIVIDADES (no recibes nada anterior)")

        def _valor(fila, *nombres, defecto=None):
            for nombre in nombres:
                if nombre in fila.index and pd.notna(fila[nombre]):
                    return fila[nombre]
            return defecto

        for _, fila in actividades_recientes.iterrows():
            km = _valor(fila, "Km", "Distancia_km", defecto=0)
            minutos = _valor(fila, "Duración Minutos", "Minutos", "Min", defecto=0)
            carga = _valor(fila, "Carga", defecto=0)
            fecha = _valor(fila, "Fecha", defecto="fecha desconocida")
            tipo = _valor(fila, "Tipo de actividad", defecto="Actividad")
            desnivel = _valor(fila, "D+ (m)", "Desnivel positivo", defecto=0)
            tiempo_total = _valor(fila, "Tiempo total")
            # Los nombres van del más reciente al más antiguo: si la tabla cambia de
            # etiqueta, _valor devuelve el defecto en silencio y el dato desaparece.
            ritmo = _valor(fila, "Ritmo min/km", "Ritmo (min:s/km)", "Ritmo (min/km)")
            fc = _valor(fila, "FC media", "Ritmo cardiaco promedio")
            zona = _valor(fila, "Zona media")
            
            if isinstance(ritmo, (int, float)):
                ritmo = f"{int(ritmo)}:{int(round((ritmo % 1) * 60)):02d}"
            ritmo_txt = f", ritmo {ritmo} min/km" if ritmo not in (None, "—") else ""
            velocidad = _valor(fila, "Velocidad km/h", "Vel (km/h)", "Velocidad (km/h)")
            vel_txt = f", velocidad {velocidad:.1f} km/h" if velocidad is not None else ""

            total_txt = f" (tiempo total con paradas: {tiempo_total})" if tiempo_total else ""

            if fc is None:
                fc_txt = ", SIN pulso registrado"
            else:
                zona_txt = f" (zona media: {zona})" if zona and zona != "—" else ""
                fc_txt = f", FC media {fc:.0f} ppm{zona_txt}"

            lineas.append(f"- {fecha}: {tipo}, {km:.1f} km en {minutos:.0f} min en movimiento"
                          f"{total_txt}, desnivel positivo {desnivel:.0f} m"
                          f"{ritmo_txt}{vel_txt}{fc_txt}, carga {carga:.0f}.")
            
        lineas.append("Si el atleta pregunta por 'mi última salida' o describe una sesión concreta, "
                      "identifícala en esta lista antes de decir que no tienes el dato.")
        
        lineas.append("El ritmo mostrado se calcula sobre distancia horizontal. En salidas con "
                      "desnivel alto no refleja el esfuerzo real: interpreta esas sesiones "
                      "considerando el desnivel y el tiempo total, no solo el ritmo.")

    lineas.append("")
    lineas.append("COMPONENTE PREDICTIVO (regresión por mesociclos)")
    if prediccion and entrenamiento:
        lineas.append(f"- Ritmo de competición estimado: {prediccion['ritmo']:.2f} min/km.")
        lineas.append(f"- Tiempo proyectado en {prediccion['distancia']:.1f} km: {prediccion['tiempo_texto']}.")
        lineas.append(f"- Fiabilidad del modelo: R²={entrenamiento['r2']:.2f}, "
                      f"error medio {entrenamiento['mae']:.2f} min/km "
                      f"(entrenado con {entrenamiento['n_mesociclos']} mesociclos).")
        lineas.append("- Limitación: el objetivo es el mejor esfuerzo del bloque, no un tiempo oficial de competencia.")
    else:
        lineas.append("- Sin predicción disponible: no hay carreras recientes suficientes en el bloque actual.")

    if diario:
        lineas.append("")
        lineas.append("DIARIO DE ESTADO FÍSICO (declarado por el atleta)")
        for registro in diario[-MAX_ENTRADAS_DIARIO:]:
            nota = f" — {registro['nota']}" if registro.get("nota") else ""
            lineas.append(f"- {registro['fecha']}: {registro['estado']}{nota}")
        lineas.append("IMPORTANTE: prioriza estos estados declarados sobre los indicadores. "
                      "Si el atleta ha estado enfermo o lesionado en los últimos días, tenlo en cuenta "
                      "aunque él proponga una sesión exigente.")

    lineas += [
        "",
        "REGLAS",
        "1. Fundamenta cada recomendación en los indicadores anteriores, citando el dato concreto.",
        "2. Nunca inventes datos que no aparezcan en este contexto.",
        "3. Recuerda que eres un apoyo a la decisión: la decisión final es del atleta.",
        "4. Si detectas riesgo de lesión o el atleta declara estar enfermo, dilo con claridad.",
        "5. Distingue siempre entre lo que mide el sistema y lo que el atleta declara en "
        "la conversación. Lo declarado se acepta como contexto, se marca como tal y no se "
        "presenta como dato verificado ni se usa para sostener una conclusión numérica.",
        "6. Si te preguntan algo que los datos disponibles no permiten confirmar, dilo con "
        "claridad y señala qué faltaría. Sin pulso registrado no se puede verificar en qué "
        "zona de intensidad se entrenó, por mucho que el atleta la haya estimado.",
        "7. La zona indicada por actividad se deriva de la FC media y resume la sesión "
        "entera. No describe el reparto real del esfuerzo, así que una sesión de series "
        "puede promediar una zona intermedia sin haber permanecido en ella.",
        "8. Si preguntan por una fecha que no aparece en la lista de actividades, di que "
        "queda fuera de la ventana que recibes y no supongas que no está sincronizada.",
        "9. La duración de una sesión es el tiempo EN MOVIMIENTO. El tiempo total "
        "incluye las paradas y solo se cita al hablar de la marca oficial de una "
        "competición, nunca como duración del entrenamiento.",
        "10. El ritmo y la velocidad vienen ya calculados en la lista de actividades. "
        "Úsalos tal cual y no los deduzcas dividiendo distancia entre tiempo.",
        "11. 'bajo Z1' significa que hay pulso pero por debajo de la zona 1, típico de "
        "caminatas o recuperación. No es lo mismo que no tener pulso.",
        "12. Este contexto se reconstruye en cada mensaje. Si lo que ves aquí contradice "
        "algo que dijiste antes, manda este contexto. Nunca digas que no tienes acceso a "
        "datos actualizados: es falso.",
    ]

    # Va al final a propósito: es lo último que lee el modelo antes de responder y lo
    # que más pesa frente a su propia respuesta anterior, que es donde se anclaba.
    if cambios:
        lineas += [
            "",
            "ATENCIÓN: DATOS MODIFICADOS DESDE TU RESPUESTA ANTERIOR",
            "El atleta cambió esto después de tu último mensaje. Lo que dijiste antes sobre "
            "estas actividades ya no vale: respóndele usando el valor nuevo.",
        ] + [f"- {cambio}" for cambio in cambios]
    return "\n".join(lineas)


def huella_actividades(actividades):
    """Resume el pulso de cada actividad para poder comparar entre un mensaje y el siguiente.

    La clave incluye la distancia porque dos sesiones del mismo deporte el mismo día,
    como un doble turno de carrera, comparten fecha y tipo.

    Args:
        actividades (pd.DataFrame): Tabla que se le pasa al agente como contexto.
    Returns:
        dict[str, int | None]: FC media redondeada por actividad, o None si no tiene pulso.
    """
    if actividades is None or actividades.empty:
        return {}
    huella = {}
    for _, fila in actividades.iterrows():
        clave = f"{fila['Fecha']} {fila['Tipo de actividad']} de {fila['Km']:.1f} km"
        fc = fila.get("FC media")
        huella[clave] = None if pd.isna(fc) else int(round(float(fc)))
    return huella


def cambios_de_pulso(previa, actual):
    """Describe las actividades cuyo pulso cambió desde el mensaje anterior.

    El modelo tiende a sostener lo que ya afirmó aunque el contexto diga otra cosa, así
    que el cambio se le entrega escrito en lugar de esperar que lo descubra comparando.

    Args:
        previa (dict): Huella del contexto que recibió el modelo en el mensaje anterior.
        actual (dict): Huella del contexto que va a recibir ahora.
    Returns:
        list[str]: Una frase por actividad modificada; vacía si no hay mensaje anterior.
    """
    cambios = []
    for clave, fc in actual.items():
        if clave not in previa or previa[clave] == fc:
            continue
        antes = previa[clave]
        if antes is None:
            cambios.append(f"{clave}: antes SIN pulso, ahora FC media {fc} ppm.")
        elif fc is None:
            cambios.append(f"{clave}: antes FC media {antes} ppm, ahora sin pulso.")
        else:
            cambios.append(f"{clave}: la FC media pasó de {antes} a {fc} ppm.")
    return cambios

def consultar_agente(cliente, contexto, mensajes, aviso=None):
    """Envía la conversación al modelo junto con el contexto del sistema.

    Args:
        cliente: Cliente de la API de Anthropic.
        contexto (str): Prompt de sistema con los datos del atleta.
        mensajes (list): Historial completo; se envían solo los más recientes.
        aviso (str | None): Cambios en los datos desde la respuesta anterior.
    Returns:
        str: Texto de la respuesta del modelo.
    """
    recientes = [{"role": m["role"], "content": m["content"]} for m in mensajes[-MAX_MENSAJES:]]
    # El aviso se pega al último mensaje del atleta solo en la copia que viaja a la API:
    # es la posición que más pesa para el modelo y así no ensucia el historial guardado.
    if aviso and recientes and recientes[-1]["role"] == "user":
        recientes[-1]["content"] += f"\n\n[Aviso del sistema, no lo escribió el atleta: {aviso}]"

    respuesta = cliente.messages.create(
        model=MODELO,
        # Acota la longitud de la respuesta del modelo, no la del prompt de entrada.
        max_tokens=2000,
        system=contexto,
        messages=recientes,
    )
    texto = respuesta.content[0].text
    # Un stop_reason de 'max_tokens' significa que la respuesta quedó incompleta.
    if respuesta.stop_reason == "max_tokens":
        texto += "\n\n_(Respuesta cortada por longitud. Pide que continúe si falta algo.)_"
    return texto