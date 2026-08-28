"""
Frontend (Streamlit) del SII "Entrenador IA FCDIA".

Capa de presentación del sistema: panel de indicadores, análisis de carga,
componente predictivo con planificador de mesociclo y agente conversacional.
Toda la lógica vive en backend.py, modelo.py, agente.py y data_loader.py.
"""

import os                                                                          # Acceso a variables de entorno
import pandas as pd
import numpy as np                                                                # Manejo puntual de datos
import plotly.express as px                                                        # Gráficos interactivos
import plotly.graph_objects as go                                                  # Trazos y bandas personalizadas
import streamlit as st                                                             # Framework de la interfaz web
import anthropic                                                                    # Cliente oficial de la API de Claude
from dotenv import load_dotenv                                                      # Carga del archivo .env local

import backend                                                                      # Procesamiento e indicadores
import modelo                                                                       # Componente predictivo
import agente                                                                       # Agente LLM y memoria
import data_loader                                                                  # Adquisición de datos

# Configuración de la API de Claude (Anthropic) para el agente conversacional
load_dotenv()                                                                       # Lee el archivo .env local y lo carga en memoria
API_KEY_CLAUDE = os.getenv("ANTHROPIC_API_KEY")                                     # Recupera de forma segura la API Key

# CONFIGURACIÓN DE LA PÁGINA

st.set_page_config(page_title=" MY Coach IA", page_icon="🏃", layout="wide")  # Configura la ventana del navegador

st.markdown(                                                                        # Inyecta estilos para las tarjetas de KPI
    """
    <style>
        div[data-testid="stMetric"] { background:#FFF; border:1px solid #E8E8E8;
            border-radius:12px; padding:14px 16px; }
        div[data-testid="stMetricValue"] { font-size:24px; }
    </style>
    """,
    unsafe_allow_html=True,                                                         # Permite renderizar CSS personalizado
)

st.title("🏃 Athletix 🚲")                          # Cabecera principal de la aplicación
st.caption("Monitoreo de carga, predicción de rendimiento y decisiones asistidas por un agente LLM.")  # Descripción breve

# Carga y procesamiento de datos históricos (actividades_strava.csv)

@st.cache_data(show_spinner="Procesando actividades...")                            # Evita recalcular en cada interacción
def obtener_datos(version):
    """Carga y procesa el histórico. El parámetro 'version' invalida la caché tras sincronizar."""
    return backend.cargar_y_procesar_datos()                                        # Delega el trabajo al backend


if "version_datos" not in st.session_state:                                         # Contador de invalidación de caché
    st.session_state.version_datos = 0                                              # Valor inicial de la versión

datos_completos = obtener_datos(st.session_state.version_datos)                     # Obtiene el DataFrame procesado

if datos_completos is None or datos_completos.empty:                                # Si no hay ninguna fuente de datos
    st.error("⚠️ No se encontró 'actividades_strava.csv'. Colócalo en la carpeta del proyecto.")  # Aviso amigable
    st.stop()                                                                       # Detiene la ejecución de la página

# Bloque lateral de controles: selección de deporte, periodo, sincronización y diario de estado

with st.sidebar:                                                                    # Panel lateral de controles
    st.header("⚙️ Controles")                                                       # Título del panel

    deportes = ["Carrera", "Bicicleta", "Todo"]                                     # Deportes seleccionables
    deporte = st.radio("Deporte", deportes, index=0)                                # Carrera como valor por defecto

    años = sorted(datos_completos["Año"].unique(), reverse=True)                    # Años presentes en el histórico
    año = st.selectbox("Periodo", ["Todo el histórico"] + [str(a) for a in años])   # Selector de periodo de análisis

    st.divider()                                                                    # Separador visual del panel
    st.subheader("🔄 Sincronización")                                               # Sección de la API de Strava

    if data_loader.credenciales_strava_disponibles():                              # Solo si el .env tiene las claves
        if st.button("Descargar actividades nuevas", width="stretch"):             # Botón de sincronización incremental
            with st.spinner("Conectando con Strava..."):                            # Indicador de proceso en curso
                ultima = datos_completos["Fecha"].max()                             # Fecha de la última actividad conocida
                nuevas, mensaje = data_loader.sincronizar_con_strava(ultima)        # Descarga solo lo posterior a esa fecha

            if nuevas > 0:                                                          # Si llegaron actividades nuevas
                st.session_state.version_datos += 1                                 # Invalida la caché de datos
                st.success(mensaje)                                                 # Informa del resultado
                st.rerun()                                                          # Recarga la interfaz con los datos nuevos
            else:                                                                   # Si no hubo novedades o hubo error
                st.info(mensaje)                                                    # Muestra el mensaje devuelto
    else:                                                                           # Si faltan credenciales en el .env
        st.caption("Añade STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET y "                # Explica cómo habilitar la función
                   "STRAVA_REFRESH_TOKEN al .env para activar la sincronización.")

    st.divider()                                                                    # Separador visual del panel
    st.subheader("🩺 Diario de estado")                                             # Sección del diario físico

    estado_hoy = st.selectbox("¿Cómo te sientes hoy?",                              # Estado físico declarado por el atleta
                              ["Normal", "Cansado", "Con dolor", "Enfermo", "En plena forma"])
    nota_hoy = st.text_input("Nota (opcional)", placeholder="Ej: molestia en el gemelo")  # Detalle libre del estado

    if st.button("Registrar estado de hoy", width="stretch"):                       # Guarda el estado en el diario
        agente.registrar_estado(estado_hoy, nota_hoy)                               # Persiste el registro en disco
        st.success("Estado registrado.")                                            # Confirma la operación al usuario

# Aplica los filtros de deporte y periodo al histórico completo para el análisis de la interfaz
datos = datos_completos.copy()                                                      # Copia sobre la que aplicar los filtros

if deporte != "Todo":                                                               # Si se ha seleccionado un deporte
    datos = datos[datos["Tipo de actividad"] == deporte]                            # Filtra por el deporte elegido
if año != "Todo el histórico":                                                      # Si se ha acotado el periodo
    datos = datos[datos["Año"] == int(año)]                                         # Filtra por el año seleccionado

if datos.empty:                                                                     # Si la combinación no devuelve nada
    st.warning("No hay actividades con esos filtros. Prueba otra combinación.")     # Avisa al usuario
    st.stop()                                                                       # Detiene el renderizado

# El diagnóstico de fatiga se calcula SIEMPRE sobre el histórico completo y con TODOS
# los deportes. El organismo acumula una única fatiga: separar la carga por disciplina
# produciría diagnósticos contradictorios (riesgo en bicicleta y óptimo en carrera a la
# vez) porque cada cálculo ignoraría la carga aportada por el otro deporte.
diagnostico = backend.estado_actual(datos_completos)                                # Estado de fatiga global vigente hoy

# Los días de inactividad sí se miden por deporte: informan de cuándo se practicó
# por última vez la disciplina seleccionada, sin alterar el diagnóstico de carga.
base_deporte = datos_completos                                                      # Punto de partida por disciplina
if deporte != "Todo":                                                               # Aplica únicamente el filtro de deporte
    base_deporte = base_deporte[base_deporte["Tipo de actividad"] == deporte]

dias_deporte = backend.dias_sin_entrenar(base_deporte)                              # Inactividad en la disciplina elegida
kpis = backend.calcular_kpis(datos)                                                 # Indicadores del periodo seleccionado
kpis_hoy = backend.calcular_kpis(base_deporte)                                      # Indicadores vigentes para el agente

# Pestñas principales de la interfaz: panel, carga y fatiga, predicción y plan, histórico
tab_panel, tab_carga, tab_pred, tab_hist = st.tabs(                                 # Estructura principal de la interfaz
    ["📊 Panel", "🔥 Carga y Fatiga", "🎯 Predicción y Plan", "📚 Histórico"]
)
# Pestaña 1: panel de indicadores y resumen del periodo
with tab_panel:                                                                     # Resumen numérico del estado actual
    st.subheader(f"Estado actual — {deporte}")                                      # Título contextualizado al deporte

    etiqueta_dias = "Días sin entrenar" if deporte == "Todo" else f"Días sin {deporte.lower()}"  # Contextualiza la métrica

    c1, c2, c3, c4 = st.columns(4)                                                  # Cuatro tarjetas de indicadores
    c1.metric("Km últimos 7 días", f"{kpis_hoy.get('km_7d', 0):.1f}")               # Volumen de la última semana
    c2.metric("Km últimos 28 días", f"{kpis_hoy.get('km_28d', 0):.1f}")             # Volumen del último mesociclo
    c3.metric(etiqueta_dias, f"{dias_deporte if dias_deporte is not None else '—'}")  # Inactividad en la disciplina

    acwr = diagnostico.get("acwr")                                                  # Ratio de fatiga vigente
    c4.metric("Ratio de fatiga (ACWR)", f"{acwr:.2f}" if pd.notna(acwr) else "—",   # Componente de análisis principal
              help="Se calcula con todos los deportes juntos: la fatiga que acumula el cuerpo es una sola.")

    if pd.isna(acwr):                                                               # Si no se pudo calcular el ratio
        st.info("No hay carga suficiente en los últimos 28 días para calcular el ACWR.")  # Explica la ausencia
    elif acwr > backend.UMBRAL_RIESGO:                                              # Zona de sobrecarga
        st.error(f"**{diagnostico['estado']}** — la carga reciente supera tu base. Reduce intensidad y prioriza recuperación.")
    elif acwr > backend.UMBRAL_OPTIMO_ALTO:                                         # Franja intermedia de vigilancia
        st.warning(f"**{diagnostico['estado']}** — vas camino de la zona de riesgo. No subas más el volumen esta semana.")
    elif acwr < backend.UMBRAL_OPTIMO_BAJO:                                         # Zona de carga insuficiente
        st.warning(f"**{diagnostico['estado']}** — puedes subir el volumen de forma gradual (máximo 10 % por semana).")
    else:                                                                           # Franja de progresión segura
        st.success(f"**{diagnostico['estado']}** — la carga está equilibrada. Mantén la progresión actual.")

    st.caption("El diagnóstico de carga considera todos los deportes en conjunto y no cambia al filtrar por disciplina.")

    st.divider()                                                                    # Separador de secciones
    st.subheader(f"Resumen del periodo — {año}")                                    # Indicadores del periodo filtrado

    # La marca de referencia se expresa en la unidad propia de cada disciplina: ritmo
    # (min/km) en carrera y velocidad media (km/h) en ciclismo. Con el filtro en "Todo"
    # se muestran ambas, ya que el ritmo por kilómetro de la bicicleta siempre sería el
    # mejor del conjunto y ocultaría la marca real de carrera.
    mejor_carrera = kpis.get("mejor_ritmo_carrera")                                 # Mejor ritmo corriendo del periodo
    mejor_bici = kpis.get("mejor_velocidad_bici")                                   # Mejor velocidad pedaleando

    if deporte == "Todo":                                                           # Ambas disciplinas a la vista
        d1, d2, d3, d4, d5 = st.columns(5)                                          # Cinco indicadores del periodo
        d3.metric("Mejor ritmo (carrera)",                                          # Marca de referencia corriendo
                  f"{backend.formatear_ritmo(mejor_carrera)} min/km")
        d4.metric("Mejor velocidad (bici)",                                         # Marca de referencia en bicicleta
                  f"{mejor_bici:.1f} km/h" if pd.notna(mejor_bici) else "—")
        col_fc = d5                                                                 # La cobertura ocupa la última columna
    else:                                                                           # Una sola disciplina seleccionada
        d1, d2, d3, col_fc = st.columns(4)                                          # Cuatro indicadores del periodo
        if deporte == "Bicicleta":                                                  # En ciclismo interesa la velocidad
            d3.metric("Mejor velocidad", f"{mejor_bici:.1f} km/h" if pd.notna(mejor_bici) else "—")
        else:                                                                       # En carrera interesa el ritmo
            d3.metric("Mejor ritmo", f"{backend.formatear_ritmo(mejor_carrera)} min/km")

    d1.metric("Actividades", f"{kpis.get('actividades', 0)}")                       # Número de sesiones registradas
    d2.metric("Distancia total", f"{kpis.get('km_total', 0):.1f} km")               # Volumen acumulado del periodo
    col_fc.metric("Cobertura de pulsómetro", f"{kpis.get('cobertura_fc', 0):.0f} %")  # Calidad real de los datos de FC

    if kpis.get("cobertura_fc", 100) < 70:                                          # Advierte si faltan muchos datos de FC
        st.caption("ℹ️ Parte del histórico no tiene frecuencia cardíaca. La carga de esas sesiones se estima "
                   "a partir de la duración en lugar de imputar pulsaciones inexistentes.")

    st.divider()                                                                    # Separador de secciones
    st.subheader("Volumen semanal")                                                 # Evolución del volumen de entrenamiento

    semanal = backend.resumen_semanal(datos)                                        # Agregación por semana y deporte
    if not semanal.empty:                                                           # Solo si hay datos agregados
        fig_sem = px.bar(semanal, x="Semana", y="Kilometros", color="Tipo de actividad",  # Barras apiladas por deporte
                         labels={"Kilometros": "Km", "Semana": ""},
                         custom_data=["Rango"])                                     # Adjunta el rango de la semana
        fig_sem.update_traces(                                                      # Etiqueta del cursor con la semana completa
            hovertemplate="Semana %{customdata[0]}<br>%{fullData.name}: %{y:.1f} km<extra></extra>")
        fig_sem.update_layout(yaxis_tickformat=".1f", hovermode="x unified",        # Etiquetas con un solo decimal
                              legend_title_text="", height=340, margin=dict(t=10))
        st.plotly_chart(fig_sem, width="stretch")                                   # Renderiza el gráfico semanal                                   # Renderiza el gráfico semanal

# Pestaña 2: análisis de carga y fatiga, con el ACWR como indicador principal
with tab_carga:                                                                     # Análisis del equilibrio de carga
    st.subheader("Fatiga frente a condición física")                                # Título de la sección
    st.caption("La carga aguda (7 días) refleja la fatiga reciente; la crónica (28 días), la condición acumulada. "
               "El cociente entre ambas es el ACWR.")                               # Explicación del indicador

    serie = backend.serie_carga_diaria(datos_completos)                             # Serie diaria de carga (todos los deportes)

    if serie.empty:                                                                 # Sin serie no hay nada que mostrar
        st.info("No hay datos suficientes para construir la serie de carga.")       # Mensaje informativo
    else:
        # La serie se calcula siempre sobre el histórico completo (las medias móviles de
        # 28 días necesitan los días previos), pero se recorta al periodo seleccionado.
        vista = serie                                                               # Serie completa por defecto
        if año != "Todo el histórico":                                              # Si se ha acotado el periodo
            vista = serie[serie["Fecha"].dt.year == int(año)]                       # Recorta al año seleccionado

        if vista.empty:                                                             # Si el año no contiene carga
            st.info("No hay carga registrada en el periodo seleccionado.")          # Mensaje informativo
        else:
            fig_eq = go.Figure()                                                    # Gráfico comparativo de ambas cargas
            fig_eq.add_trace(go.Scatter(                                            # Trazo de la condición física
                x=vista["Fecha"], y=vista["Carga Cronica"], name="Condición",
                line=dict(color="#2E86DE", width=2, shape="spline", smoothing=1.3),  # Línea suavizada
                hovertemplate="Condición: %{y:.1f}<extra></extra>"))                # Etiqueta breve para el recuadro
            fig_eq.add_trace(go.Scatter(                                            # Trazo de la fatiga reciente
                x=vista["Fecha"], y=vista["Carga Aguda"], name="Fatiga",
                line=dict(color="#EE5A24", width=2, shape="spline", smoothing=1.3),  # Línea suavizada
                hovertemplate="Fatiga: %{y:.1f}<extra></extra>"))                   # Etiqueta breve para el recuadro
            fig_eq.update_layout(height=330, hovermode="x unified", margin=dict(t=10),  # Ajustes de presentación
                                 yaxis_title="Carga media diaria", xaxis_title="",
                                 yaxis_tickformat=".1f", legend_title_text="")
            st.plotly_chart(fig_eq, width="stretch")                                # Renderiza el gráfico de equilibrio
            st.caption("Fatiga = carga media de los últimos 7 días. Condición = carga media de los últimos 28 días. "
                       "Cuando la naranja supera a la azul, estás acumulando más fatiga de la que tu base soporta.")

            st.subheader("Evolución del ACWR")                                      # Gráfico del ratio agudo:crónico

            fig_acwr = go.Figure()                                                  # Gráfico del ratio con bandas
            fig_acwr.add_hrect(y0=backend.UMBRAL_OPTIMO_BAJO, y1=backend.UMBRAL_OPTIMO_ALTO,  # Banda verde recomendada
                               fillcolor="#2ECC71", opacity=0.12, line_width=0,
                               annotation_text="Zona óptima", annotation_position="top left")
            fig_acwr.add_hrect(y0=backend.UMBRAL_OPTIMO_ALTO, y1=backend.UMBRAL_RIESGO,  # Banda ámbar de vigilancia
                               fillcolor="#F39C12", opacity=0.10, line_width=0)
            fig_acwr.add_trace(go.Scatter(                                          # Trazo del ratio a lo largo del tiempo
                x=vista["Fecha"], y=vista["ACWR"], name="ACWR",
                line=dict(color="#333", width=2, shape="spline", smoothing=1.3),    # Línea suavizada
                hovertemplate="ACWR: %{y:.2f}<extra></extra>"))                     # Etiqueta breve para el recuadro
            fig_acwr.add_hline(y=backend.UMBRAL_RIESGO, line_dash="dash", line_color="#E74C3C")  # Umbral de riesgo
            fig_acwr.add_hline(y=backend.UMBRAL_OPTIMO_BAJO, line_dash="dash", line_color="#F39C12")  # Umbral inferior
            fig_acwr.update_layout(height=330, margin=dict(t=10), yaxis_title="ACWR",  # Ajustes de presentación
                                   xaxis_title="", yaxis_tickformat=".1f", yaxis_range=[0, 2.5])
            st.plotly_chart(fig_acwr, width="stretch")                              # Renderiza el gráfico del ACWR

# Pestaña 3: componente predictivo de rendimiento y planificador de mesociclo
with tab_pred:                                                                      # Componente predictivo del sistema
    st.subheader("Predicción de rendimiento por mesociclo")                         # Título de la sección
    st.caption("Regresión lineal múltiple entrenada sobre bloques de 28 días. Se usan mesociclos y no semanas "
               "porque el descanso previo a competición (tapering) sesgaría el modelo hacia tiempos más lentos.")

    entrenamiento = modelo.entrenar_modelo(datos_completos)                         # Entrena con todo el histórico disponible

    if entrenamiento is None:                                                       # Sin muestra suficiente para modelar
        st.warning("No hay mesociclos suficientes con carreras de 5 km o más para entrenar el modelo.")  # Aviso honesto
    else:
        m1, m2, m3 = st.columns(3)                                                  # Métricas de fiabilidad del modelo
        m1.metric("Mesociclos de entrenamiento", entrenamiento["n_mesociclos"])     # Tamaño real de la muestra
        m2.metric("R² (validación LOO)", f"{entrenamiento['r2']:.2f}")              # Bondad de ajuste fuera de muestra
        m3.metric("Error medio", f"{entrenamiento['mae']:.2f} min/km")              # Error absoluto medio del modelo

        mejora = 100 * (1 - entrenamiento["mae"] / entrenamiento["mae_baseline"])   # Mejora frente a predecir la media
        st.caption(f"El modelo reduce el error un {mejora:.0f} % frente a predecir siempre tu ritmo medio "
                   f"({entrenamiento['mae_baseline']:.2f} min/km). Validado con Leave-One-Out sobre datos reales.")

        st.divider()                                                                # Separador de secciones
        metricas = modelo.metricas_ultimo_bloque(datos_completos)                   # Métricas del mesociclo vigente

        if metricas is None:                                                        # Sin carreras recientes no hay predicción
            st.info("No hay carreras de 5 km o más en los últimos 28 días. Sal a correr y vuelve a consultar.")
            prediccion = None                                                       # Sin predicción disponible
        else:
            col_dist, col_res = st.columns([1, 2])                                  # Columna de control y de resultado

            with col_dist:                                                          # Selector de la distancia objetivo
                distancia = st.selectbox("Distancia objetivo", [5.0, 10.0, 15.0, 21.1],
                                         index=1, format_func=lambda d: f"{d:g} km")

            ritmo = modelo.predecir_ritmo(entrenamiento, metricas)                  # Ritmo de competición estimado
            tiempo = modelo.ritmo_a_tiempo(ritmo, distancia)                        # Tiempo total mediante Riegel
            tiempo_texto = modelo.formatear_tiempo(tiempo)                          # Formato de cronómetro legible

            with col_res:                                                           # Presentación del resultado predictivo
                r1, r2 = st.columns(2)                                              # Dos indicadores de la predicción
                r1.metric(f"Tiempo estimado en {distancia:g} km", tiempo_texto)     # Marca proyectada por el modelo
                r2.metric("Ritmo de competición",                                   # Ritmo objetivo por kilómetro
                          f"{backend.formatear_ritmo(ritmo)} min/km")

            if modelo.fuera_de_rango_calibrado(distancia):                          # Advierte de la extrapolación
                st.warning(f"⚠️ El modelo está calibrado con esfuerzos de {modelo.RANGO_CALIBRADO[0]} a "
                           f"{modelo.RANGO_CALIBRADO[1]} km. La proyección a {distancia:g} km es una extrapolación "
                           "corregida con la fórmula de Riegel y su fiabilidad es menor.")

            prediccion = {"ritmo": ritmo, "distancia": distancia, "tiempo_texto": tiempo_texto}  # Contexto para el agente

            st.divider()                                                            # Separador de secciones
            st.subheader("🎯 Planificador de mesociclo")                            # Uso inverso del modelo
            st.caption("El modelo se invierte: en lugar de predecir tu marca a partir del entrenamiento, "
                       "despeja el volumen semanal necesario para alcanzar la marca que te propongas.")

            p1, p2 = st.columns([1, 2])                                             # Control de meta y resultado del plan

            with p1:                                                                # Entrada de la marca objetivo
                meta_min = st.number_input(f"Meta en {distancia:g} km (minutos)",
                                           min_value=10.0, max_value=300.0,
                                           value=float(round(tiempo * 0.95, 1)), step=1.0)

            ritmo_meta_ref = (meta_min / ((distancia / 10.0) ** modelo.EXPONENTE_RIEGEL)) / 10.0  # Riegel inverso a ritmo base
            plan = modelo.planificar_volumen(entrenamiento, metricas, ritmo_meta_ref)  # Despeja el volumen requerido

            with p2:                                                                # Resultado del planificador
                if not plan["viable"]:                                              # Meta inalcanzable o modelo sin señal
                    st.error(plan["mensaje"])                                       # Explica por qué no es viable
                else:
                    q1, q2 = st.columns(2)                                          # Comparativa de volumen actual y objetivo
                    q1.metric("Volumen actual", f"{plan['km_actual']:.1f} km/sem")  # Carga semanal vigente
                    q2.metric("Volumen necesario", f"{plan['km_necesarios']:.1f} km/sem",
                              delta=f"{plan['incremento_pct']:+.0f} %")             # Carga requerida y variación exigida

                    if plan["riesgo"]:                                              # Incremento superior al 10 % semanal
                        st.error(f"⚠️ Ese salto de volumen ({plan['incremento_pct']:+.0f} %) supera la regla del 10 % "
                                 "semanal y te llevaría a la zona de riesgo de lesión del ACWR. "
                                 "Reparte el incremento en varios mesociclos.")
                    else:                                                           # Progresión dentro de lo recomendable
                        st.success("✅ El incremento necesario está dentro de una progresión segura (menos del 10 %).")

        st.divider()                                                                # Separador de secciones
        st.subheader("Progresión: mejor ritmo por mesociclo")                       # Serie histórica del rendimiento

        historico = entrenamiento["datos"].reset_index().rename(columns={"index": "Mesociclo"})  # Dataset de entrenamiento

        # El número de bloque no es interpretable por sí solo: se traduce a la fecha real
        # de inicio del mesociclo, contada desde la primera actividad del histórico.
        origen = datos_completos["Fecha"].min().normalize()                         # Fecha de la primera actividad
        historico["Inicio"] = origen + pd.to_timedelta(historico["Mesociclo"] * 28, unit="D")  # Inicio de cada bloque

        fig_prog = px.line(historico, x="Inicio", y="mejor_ritmo", markers=True,    # Evolución del mejor esfuerzo
                           labels={"mejor_ritmo": "Mejor ritmo (min/km)", "Inicio": "Inicio del bloque"})
        
        historico["Ritmo texto"] = historico["mejor_ritmo"].apply(backend.formatear_ritmo)  # Ritmo en formato min:seg
        fig_prog.update_traces(customdata=historico[["Ritmo texto"]],               # Adjunta el ritmo ya formateado
                               hovertemplate="Bloque iniciado el %{x|%d/%m/%Y}<br>"
                                             "Mejor ritmo: %{customdata[0]} min/km<extra></extra>")

        minimo = int(np.floor(historico["mejor_ritmo"].min()))                      # Minuto entero inferior de la serie
        maximo = int(np.ceil(historico["mejor_ritmo"].max()))                       # Minuto entero superior de la serie
        marcas = list(range(minimo, maximo + 1))                                    # Escala de minutos enteros: 5, 6, 7, 8

        fig_prog.update_yaxes(autorange="reversed", tickmode="array",               # Fuerza el uso de las marcas indicadas
                              tickvals=marcas, ticktext=[str(m) for m in marcas])   # Etiquetas sin decimales
        
        minimo = int(np.floor(historico["mejor_ritmo"].min()))                      # Minuto entero inferior de la serie
        maximo = int(np.ceil(historico["mejor_ritmo"].max()))                       # Minuto entero superior de la serie
        marcas = list(range(minimo, maximo + 1))                                    # Escala de minutos enteros: 5, 6, 7, 8
        
        fig_prog.update_yaxes(autorange="reversed", tickvals=marcas,                # Invierte el eje: más abajo es más rápido
                              ticktext=[str(m) for m in marcas])                    # Etiquetas sin decimales
        
        fig_prog.update_layout(height=330, margin=dict(t=10))                       # Ajustes de presentación
        st.plotly_chart(fig_prog, width="stretch")                                  # Renderiza la curva de progresión
        st.caption(f"Cada punto es un bloque de 28 días, contados desde la primera actividad "
                   f"({origen.strftime('%d/%m/%Y')}). Solo aparecen los bloques con al menos una carrera de 5 km o más. "
                   "El eje está invertido: cuanto más abajo el ritmo, más rápido.")

# Pestaña 4: histórico de actividades, con resumen y detalle de las sesiones recientes
with tab_hist:                                                                      # Consulta del historial de actividades
    st.subheader("Resumen por deporte")                                             # Distribución global de la práctica

    resumen = backend.resumen_por_tipo(datos_completos)                              # Agregación por tipo de actividad
    if not resumen.empty:                                                           # Solo si hay datos que mostrar
        st.dataframe(resumen, width="stretch", hide_index=True)                     # Tabla resumen sin índice

    st.divider()                                                                    # Separador de secciones
    st.subheader(f"Distribución de distancias — {deporte} ({año})")                 # Histograma del deporte y periodo

    fig_hist = px.histogram(datos, x="Distancia_km", nbins=25,                      # Histograma del periodo filtrado
                            labels={"Distancia_km": "Distancia (km)"},
                            color_discrete_sequence=["#EE5A24"])
    fig_hist.update_traces(hovertemplate="Distancia: %{x:.1f} km<br>No. actividades: %{y}<extra></extra>")
    fig_hist.update_layout(height=300, margin=dict(t=10), yaxis_title="No. actividades",  # Ajustes de presentación
                           xaxis_tickformat=".1f", bargap=0.05)
    st.plotly_chart(fig_hist, width="stretch")                                      # Renderiza el histograma
    st.caption(f"Distancias de tus {len(datos)} actividades de {deporte.lower()} en el periodo seleccionado. "
               "Sirve para ver a qué distancias entrenas habitualmente.")

    st.divider()                                                                    # Separador de secciones
    st.subheader("Últimas actividades")                                             # Detalle de las sesiones recientes
    st.dataframe(backend.ultimas_actividades(datos, 10), width="stretch", hide_index=True)  # Tabla de las 10 últimas

# Agente conversacional: recibe los indicadores, la predicción y el diario de estado, y devuelve recomendaciones
st.divider()                                                                        # Separador del bloque conversacional
st.subheader("🤖 Entrenador IA")                                                    # Título de la sección del agente

if "mensajes" not in st.session_state:                                              # Primera carga de la sesión
    st.session_state.mensajes = agente.cargar_historial()                           # Recupera el historial guardado en disco

cabecera, boton = st.columns([4, 1])                                                # Cabecera y control de la conversación
cabecera.caption("El agente recibe tus indicadores, la predicción del modelo y tu diario de estado. "
                 "La conversación se guarda en disco y sobrevive al cierre de la aplicación.")

if boton.button("🗑️ Borrar memoria", width="stretch"):                              # Limpieza del historial conversacional
    agente.borrar_historial()                                                       # Elimina el fichero de memoria
    st.session_state.mensajes = []                                                  # Vacía el historial de la sesión
    st.rerun()                                                                      # Recarga la interfaz sin mensajes

# La conversación se confina a un contenedor de altura fija con desplazamiento propio,
# para que el historial no alargue la página. Por defecto solo se muestran los últimos
# intercambios, de modo que la pregunta más reciente quede siempre a la vista.
MENSAJES_VISIBLES = 6                                                               # Mensajes mostrados por defecto
total_mensajes = len(st.session_state.mensajes)                                     # Tamaño del historial completo

ver_todo = st.checkbox(f"Ver conversación completa ({total_mensajes} mensajes)",     # Permite recuperar el historial
                       value=False, disabled=total_mensajes <= MENSAJES_VISIBLES)

visibles = st.session_state.mensajes if ver_todo else st.session_state.mensajes[-MENSAJES_VISIBLES:]

ventana_chat = st.container(height=420)                                             # Contenedor con barra de desplazamiento
with ventana_chat:                                                                  # Todo el chat vive dentro de la ventana
    for mensaje in visibles:                                                        # Recorre los mensajes a mostrar
        with st.chat_message(mensaje["role"]):                                      # Abre el globo correspondiente
            st.markdown(mensaje["content"])                                         # Muestra el contenido del mensaje

if prompt := st.chat_input("Ej: ¿Qué entreno mañana? ¿Voy bien para bajar de 50 min en 10K?"):  # Entrada del usuario

    st.session_state.mensajes.append({"role": "user", "content": prompt})           # Añade el mensaje al historial
    with ventana_chat:                                                              # Escribe dentro de la ventana con scroll
        with st.chat_message("user"):                                               # Abre el globo del usuario
            st.markdown(prompt)                                                     # Renderiza el texto introducido

    if not API_KEY_CLAUDE:                                                          # Verifica que exista la clave de Claude
        st.error("❌ No se encontró 'ANTHROPIC_API_KEY' en el archivo .env local.")  # Aviso de configuración
    else:
        try:
            cliente = anthropic.Anthropic(api_key=API_KEY_CLAUDE)                   # Inicializa el cliente con la clave del .env

            entrenamiento_ctx = modelo.entrenar_modelo(datos_completos)             # Modelo para el contexto del agente
            metricas_ctx = modelo.metricas_ultimo_bloque(datos_completos)           # Métricas del mesociclo vigente
            prediccion_ctx = None                                                   # Predicción a inyectar en el contexto

            if entrenamiento_ctx and metricas_ctx:                                  # Solo si el modelo puede predecir
                ritmo_ctx = modelo.predecir_ritmo(entrenamiento_ctx, metricas_ctx)  # Ritmo estimado de competición
                prediccion_ctx = {                                                  # Empaqueta la predicción para el agente
                    "ritmo": ritmo_ctx,
                    "distancia": 10.0,                                              # Distancia de referencia estándar
                    "tiempo_texto": modelo.formatear_tiempo(modelo.ritmo_a_tiempo(ritmo_ctx, 10.0)),
                }

            contexto = agente.construir_contexto(                                   # Ensambla el prompt de sistema
                kpis_hoy, diagnostico, prediccion_ctx, entrenamiento_ctx, agente.cargar_diario(),
                actividades_recientes=backend.ultimas_actividades(datos_completos, 5),  # Detalle de las 5 últimas sesiones
            )

            with ventana_chat:                                                      # Escribe dentro de la ventana con scroll
                with st.chat_message("assistant"):                                  # Abre el globo del asistente
                    with st.spinner("Analizando tus datos..."):                     # Indicador de proceso en curso
                        texto = agente.consultar_agente(cliente, contexto, st.session_state.mensajes)  # Consulta al modelo
                    st.markdown(texto)                                              # Muestra la respuesta generada

            st.session_state.mensajes.append({"role": "assistant", "content": texto})  # Guarda la respuesta en el historial
            agente.guardar_historial(st.session_state.mensajes)                      # Persiste la conversación en disco

        except Exception as error:                                                  # Captura cualquier fallo de la API
            st.error(f"Error de conexión con la IA: {error}")                        # Muestra el error de forma amigable