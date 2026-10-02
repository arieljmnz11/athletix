"""
Frontend del SII Athletix.

Capa de presentación del sistema: panel de indicadores, análisis de carga,
componente predictivo con planificador de mesociclo, carga manual de pulso y
agente conversacional. Toda la lógica vive en backend.py, modelo.py, agente.py,
data_loader.py y metricas_fc.py.
"""

import os
from datetime import date
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import anthropic
import config
from dotenv import load_dotenv

import backend
import modelo
import agente
import data_loader
import metricas_fc
import ajustes

# En la nube las llaves llegan por Streamlit Secrets; load_dotenv solo aplica en local.
load_dotenv()
API_KEY_CLAUDE = os.getenv("ANTHROPIC_API_KEY")

# Paleta de las zonas de intensidad, de la más suave a la más exigente.
COLORES_ZONA = ["#74B9FF", "#55EFC4", "#FFEAA7", "#FAB1A0", "#EE5A24"]

st.set_page_config(page_title="Athletix", page_icon="🏃", layout="wide")

st.markdown(
    """
    <style>
        div[data-testid="stMetric"] { background:#FFF; border:1px solid #E8E8E8;
            border-radius:12px; padding:14px 16px; }
        div[data-testid="stMetricValue"] { font-size:24px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏃 Athletix 🚲")
st.caption("Monitoreo de carga, predicción de rendimiento y decisiones asistidas por un agente LLM.")


def etiqueta_actividad(clave, tipo, tiene_serie=False):
    """Convierte la clave ISO de una actividad en una etiqueta legible para el selector.

    Args:
        clave (str): Marca temporal de la actividad tal como la guarda Strava.
        tipo (str): Deporte practicado.
        tiene_serie (bool): Indica si ya hay pulso cargado a mano.
    Returns:
        str: Etiqueta con fecha, hora, deporte y marca de serie existente.
    """
    fecha = pd.to_datetime(clave, errors="coerce", utc=True)
    if pd.isna(fecha):
        texto = str(clave)
    else:
        # La API etiqueta la hora local con sufijo Z; descartar la zona conserva la hora real.
        texto = fecha.tz_localize(None).strftime("%d/%m/%Y %H:%M")
    marca = " · pulso cargado" if tiene_serie else ""
    return f"{texto} — {tipo}{marca}"


# El ttl acota cuánto puede envejecer la caché si los datos cambian desde otra
# sesión; el .clear() explícito de cada escritura fuerza el refresco inmediato.
@st.cache_data(ttl=600, show_spinner="Procesando actividades...")
def obtener_datos(version):
    """Carga y procesa el histórico. El parámetro 'version' invalida la caché tras sincronizar."""
    return backend.cargar_y_procesar_datos()


if "version_datos" not in st.session_state:
    st.session_state.version_datos = 0

datos_completos = obtener_datos(st.session_state.version_datos)

if datos_completos is None or datos_completos.empty:
    st.error("No hay actividades disponibles. Revisa que 'actividades_strava.csv' esté en el "
             "repositorio o sincroniza con Strava desde el panel lateral.")
    st.stop()

# Los ajustes se leen una sola vez por recarga y se reparten desde aquí, para que
# ninguna pestaña se invente su propia frecuencia cardíaca máxima.
ajustes_usuario = ajustes.leer()
fc_maxima_activa, origen_fcmax = ajustes.fc_maxima_efectiva(ajustes_usuario)

limites_zonas = ajustes.rangos_de_zonas(ajustes_usuario)

with st.sidebar:
    st.header("⚙️ Controles")

    deportes = ["Carrera", "Bicicleta", "Todo"]
    deporte = st.radio(
        "Deporte", deportes,
        index=ajustes.indice_por_defecto(ajustes_usuario, "deporte_defecto", deportes))

    años = sorted(datos_completos["Año"].unique(), reverse=True)
    opciones_año = ["Todo el histórico"] + [str(a) for a in años]
    año = st.selectbox(
        "Periodo", opciones_año,
        index=ajustes.indice_por_defecto(ajustes_usuario, "periodo_defecto", opciones_año))

    st.divider()
    st.subheader("🔄 Sincronización")

    if data_loader.credenciales_strava_disponibles():
        if st.button("Descargar actividades nuevas", width="stretch"):
            with st.spinner("Conectando con Strava..."):
                ultima = datos_completos["Fecha"].max()
                nuevas, mensaje = data_loader.sincronizar_con_strava(ultima)

            if nuevas > 0:
                obtener_datos.clear()
                st.session_state.version_datos += 1
                st.success(mensaje)
                st.rerun()
            else:
                st.info(mensaje)
    else:
        st.caption("Faltan las credenciales de Strava. Añádelas al .env en local o a los "
                   "Secrets de Streamlit Cloud para activar la sincronización.")

    st.divider()
    st.subheader("🩺 Diario de estado")

    estado_hoy = st.selectbox("¿Cómo te sientes hoy?",
                              ["Normal", "Cansado", "Con dolor", "Enfermo", "En plena forma"])
    nota_hoy = st.text_input("Nota (opcional)", placeholder="Ej: molestia en el gemelo")

    if st.button("Registrar estado de hoy", width="stretch"):
        agente.registrar_estado(estado_hoy, nota_hoy,
                                ajustes_usuario["retencion_diario"])
        st.success("Estado registrado.")

datos = datos_completos.copy()

if deporte != "Todo":
    datos = datos[datos["Tipo de actividad"] == deporte]
if año != "Todo el histórico":
    datos = datos[datos["Año"] == int(año)]

if datos.empty:
    st.warning("No hay actividades con esos filtros. Prueba otra combinación.")
    st.stop()

# El diagnóstico de fatiga se calcula SIEMPRE sobre el histórico completo y con TODOS
# los deportes. El organismo acumula una única fatiga: separar la carga por disciplina
# produciría diagnósticos contradictorios (riesgo en bicicleta y óptimo en carrera a la
# vez) porque cada cálculo ignoraría la carga aportada por el otro deporte.
diagnostico = backend.estado_actual(datos_completos)

# Los días de inactividad sí se miden por deporte: informan de cuándo se practicó
# por última vez la disciplina seleccionada, sin alterar el diagnóstico de carga.
base_deporte = datos_completos
if deporte != "Todo":
    base_deporte = base_deporte[base_deporte["Tipo de actividad"] == deporte]

dias_deporte = backend.dias_sin_entrenar(base_deporte)
kpis = backend.calcular_kpis(datos)
kpis_hoy = backend.calcular_kpis(base_deporte)

tab_panel, tab_carga, tab_pred, tab_pulso, tab_hist, tab_agente, tab_ajustes = st.tabs(
    ["📊 Panel", "🔥 Carga y Fatiga", "🎯 Predicción y Plan", "❤️ Pulso manual",
     "📚 Histórico", "🤖 Coach AI", "⚙️ Ajustes"]
)

with tab_panel:
    st.subheader(f"Estado actual — {deporte}")

    etiqueta_dias = "Días sin entrenar" if deporte == "Todo" else f"Días sin {deporte.lower()}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Km últimos 7 días", f"{kpis_hoy.get('km_7d', 0):.1f}")
    c2.metric("Km últimos 28 días", f"{kpis_hoy.get('km_28d', 0):.1f}")
    c3.metric(etiqueta_dias, f"{dias_deporte if dias_deporte is not None else '—'}")

    acwr = diagnostico.get("acwr")
    c4.metric("Ratio de fatiga (ACWR)", f"{acwr:.2f}" if pd.notna(acwr) else "—",
              help="Se calcula con todos los deportes juntos: la fatiga que acumula el cuerpo es una sola.")

    if pd.isna(acwr):
        st.info("No hay carga suficiente en los últimos 28 días para calcular el ACWR.")
    elif acwr > backend.UMBRAL_RIESGO:
        st.error(f"**{diagnostico['estado']}** — la carga reciente supera tu base. Reduce intensidad y prioriza recuperación.")
    elif acwr > backend.UMBRAL_OPTIMO_ALTO:
        st.warning(f"**{diagnostico['estado']}** — vas camino de la zona de riesgo. No subas más el volumen esta semana.")
    elif acwr < backend.UMBRAL_OPTIMO_BAJO:
        st.warning(f"**{diagnostico['estado']}** — puedes subir el volumen de forma gradual (máximo 10 % por semana).")
    else:
        st.success(f"**{diagnostico['estado']}** — la carga está equilibrada. Mantén la progresión actual.")

    st.caption("El diagnóstico de carga considera todos los deportes en conjunto y no cambia al filtrar por disciplina.")

    st.divider()
    st.subheader(f"Resumen del periodo — {año}")

    # La marca de referencia se expresa en la unidad propia de cada disciplina: ritmo
    # (min/km) en carrera y velocidad media (km/h) en ciclismo. Con el filtro en "Todo"
    # se muestran ambas, ya que el ritmo por kilómetro de la bicicleta siempre sería el
    # mejor del conjunto y ocultaría la marca real de carrera.
    mejor_carrera = kpis.get("mejor_ritmo_carrera")
    mejor_bici = kpis.get("mejor_velocidad_bici")

    if deporte == "Todo":
        d1, d2, d3, d4, d5 = st.columns(5)
        d3.metric("Mejor ritmo (carrera)",
                  f"{backend.formatear_ritmo(mejor_carrera)} min/km")
        d4.metric("Mejor velocidad (bici)",
                  f"{mejor_bici:.1f} km/h" if pd.notna(mejor_bici) else "—")
        col_fc = d5
    else:
        d1, d2, d3, col_fc = st.columns(4)
        if deporte == "Bicicleta":
            d3.metric("Mejor velocidad", f"{mejor_bici:.1f} km/h" if pd.notna(mejor_bici) else "—")
        else:
            d3.metric("Mejor ritmo", f"{backend.formatear_ritmo(mejor_carrera)} min/km")

    d1.metric("Actividades", f"{kpis.get('actividades', 0)}")
    d2.metric("Distancia total", f"{kpis.get('km_total', 0):.1f} km")
    col_fc.metric("Cobertura de pulsómetro", f"{kpis.get('cobertura_fc', 0):.0f} %")

    if kpis.get("cobertura_fc", 100) < 70:
        st.caption("ℹ️ Parte del histórico no tiene frecuencia cardíaca. La carga de esas sesiones se estima "
                   "a partir de la duración en lugar de imputar pulsaciones inexistentes. Puedes completar "
                   "sesiones concretas desde la pestaña «Pulso manual».")

    st.divider()
    st.subheader("Volumen semanal")

    semanal = backend.resumen_semanal(datos)
    if not semanal.empty:

        semanal["Duracion_txt"] = (semanal["Minutos"] * 60).apply(backend.formatear_duracion_corta)

        paleta = px.colors.qualitative.Plotly

        fig_sem = go.Figure()
        for indice, tipo in enumerate(sorted(semanal["Tipo de actividad"].unique())):
            sub = semanal[semanal["Tipo de actividad"] == tipo]
            # Los deportes estáticos no acumulan distancia ni desnivel, así que su línea
            # muestra la duración, que es lo único interpretable en esas sesiones.
            if tipo in backend.SIN_DISTANCIA:
                plantilla = f"{tipo}: %{{customdata[1]}}<extra></extra>"
            else:
                plantilla = (f"{tipo}: %{{y:.1f}} km<br>"
                             "Desnivel: %{customdata[0]:,.0f} m<extra></extra>")

            fig_sem.add_trace(go.Bar(
                name=tipo, x=sub["Semana"], y=sub["Kilometros"],
                marker_color=paleta[indice % len(paleta)],
                customdata=np.stack([sub["Desnivel"], sub["Duracion_txt"]], axis=-1),
                hovertemplate=plantilla))

        # El eje vuelve a ser temporal para que Plotly rotule los meses por sí solo. El
        # encabezado del hover se toma de la etiqueta del eje, así que se le da formato
        # con hoverformat en vez de con etiquetas propias, que lo dejarían inconsistente.
        fig_sem.update_layout(barmode="stack", hovermode="x unified", height=340,
                              margin=dict(t=10), legend_title_text="",
                              xaxis_title="", yaxis_title="Km",
                              xaxis_hoverformat="Semana del %d/%m/%Y",
                              yaxis_tickformat=".1f")
        evento_semana = st.plotly_chart(fig_sem, width="stretch", on_select="rerun",
                                        selection_mode="points", key="grafico_semanal")

        # Streamlit no recibe el paso del cursor desde Plotly, solo clics y selecciones, así
        # que el panel sigue a la barra que se toca. Sin selección muestra la semana más reciente.
        semanas = semanal["Semana"].dt.normalize()
        semana_activa = semanas.max()
        puntos = (evento_semana or {}).get("selection", {}).get("points", [])
        if puntos:
            tocada = pd.to_datetime(puntos[0].get("x"), errors="coerce")
            if pd.notna(tocada) and tocada.normalize() in set(semanas):
                semana_activa = tocada.normalize()

        de_la_semana = semanal[semanas == semana_activa]
        fin_sem = semana_activa + pd.Timedelta(days=6)

        st.markdown(f"**Semana del {semana_activa:%d/%m} al {fin_sem:%d/%m/%Y}**")
        s1, s2, s3 = st.columns(3)
        s1.metric("Distancia", f"{de_la_semana['Kilometros'].sum():,.1f} km")
        s2.metric("Desnivel positivo", f"{de_la_semana['Desnivel'].sum():,.0f} m")
        s3.metric("Tiempo en movimiento",
                  backend.formatear_duracion_corta(de_la_semana["Minutos"].sum() * 60))
        st.caption("Toca una barra para ver los totales de esa semana. Sin ninguna barra tocada "
                   "se muestra la más reciente.")

with tab_carga:
    st.subheader("Fatiga frente a condición física")
    st.caption("La carga aguda (7 días) refleja la fatiga reciente; la crónica (28 días), la condición acumulada. "
               "El cociente entre ambas es el ACWR.")

    serie = backend.serie_carga_diaria(datos_completos)

    if serie.empty:
        st.info("No hay datos suficientes para construir la serie de carga.")
    else:
        # La serie se calcula siempre sobre el histórico completo (las medias móviles de
        # 28 días necesitan los días previos), pero se recorta al periodo seleccionado.
        vista = serie
        if año != "Todo el histórico":
            vista = serie[serie["Fecha"].dt.year == int(año)]

        if vista.empty:
            st.info("No hay carga registrada en el periodo seleccionado.")
        else:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=vista["Fecha"], y=vista["Carga Cronica"], name="Condición",
                line=dict(color="#2E86DE", width=2, shape="spline", smoothing=1.3),
                hovertemplate="Condición: %{y:.1f}<extra></extra>"))
            fig_eq.add_trace(go.Scatter(
                x=vista["Fecha"], y=vista["Carga Aguda"], name="Fatiga",
                line=dict(color="#EE5A24", width=2, shape="spline", smoothing=1.3),
                hovertemplate="Fatiga: %{y:.1f}<extra></extra>"))
            fig_eq.update_layout(height=330, hovermode="x unified", margin=dict(t=10),
                                 yaxis_title="Carga media diaria", xaxis_title="",
                                 yaxis_tickformat=".1f", legend_title_text="")
            st.plotly_chart(fig_eq, width="stretch")
            st.caption("Fatiga = carga media de los últimos 7 días. Condición = carga media de los últimos 28 días. "
                       "Cuando la naranja supera a la azul, estás acumulando más fatiga de la que tu base soporta.")

            st.subheader("Evolución del ACWR")

            fig_acwr = go.Figure()
            fig_acwr.add_hrect(y0=backend.UMBRAL_OPTIMO_BAJO, y1=backend.UMBRAL_OPTIMO_ALTO,
                               fillcolor="#2ECC71", opacity=0.12, line_width=0,
                               annotation_text="Zona óptima", annotation_position="top left")
            fig_acwr.add_hrect(y0=backend.UMBRAL_OPTIMO_ALTO, y1=backend.UMBRAL_RIESGO,
                               fillcolor="#F39C12", opacity=0.10, line_width=0)
            fig_acwr.add_trace(go.Scatter(
                x=vista["Fecha"], y=vista["ACWR"], name="ACWR",
                line=dict(color="#333", width=2, shape="spline", smoothing=1.3),
                hovertemplate="ACWR: %{y:.2f}<extra></extra>"))
            fig_acwr.add_hline(y=backend.UMBRAL_RIESGO, line_dash="dash", line_color="#E74C3C")
            fig_acwr.add_hline(y=backend.UMBRAL_OPTIMO_BAJO, line_dash="dash", line_color="#F39C12")
            fig_acwr.update_layout(height=330, margin=dict(t=10), yaxis_title="ACWR",
                                   xaxis_title="", yaxis_tickformat=".1f", yaxis_range=[0, 2.5])
            st.plotly_chart(fig_acwr, width="stretch")

# Bloque completo de la pestaña Predicción y Plan. Reemplaza desde "with tab_pred:" hasta,
# sin incluirla, la línea "with tab_pulso:".

def segundos_texto(min_km):
    """Convierte una diferencia de ritmo en segundos por kilómetro legibles."""
    segundos = round(abs(min_km) * 60)
    if segundos == 0:
        return "sin diferencia"
    return f"{segundos} s por km " + ("más rápido" if min_km > 0 else "más lento")


with tab_pred:
    st.subheader("Predicción de rendimiento por mesociclo")
    st.caption("Regresión simple validada con Leave-One-Out donde el mejor ritmo del mesociclo "
               "anterior predice el de este bloque, con un ajuste que acerca los meses "
               "excepcionales a tu nivel habitual. Se usan mesociclos y no semanas porque el "
               "descanso previo a competición, el tapering, sesgaría el modelo hacia tiempos más lentos.")

    entrenamiento = modelo.entrenar_modelo(datos_completos)

    if entrenamiento is None:
        st.warning("No hay mesociclos suficientes con carreras de 5 km o más para entrenar el modelo.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Mesociclos de entrenamiento", entrenamiento["n_mesociclos"])
        m2.metric("R² - validación LOO", f"{entrenamiento['r2']:.2f}",
                  help="Mide cuánto de la variación entre tus bloques logra explicar el modelo "
                       "cuando predice cada uno sin haberlo visto. Vale 1 si acierta siempre. "
                       "Con 0 acierta lo mismo que repetir siempre tu ritmo medio, y por debajo "
                       "de 0 lo hace peor. La validación LOO aparta un bloque, entrena con todos "
                       "los demás, predice el apartado y repite el proceso con cada bloque.")
        m3.metric("Error medio del modelo", f"{entrenamiento['mae']:.2f} min/km",
                  help=f"Cuánto se equivoca el modelo en promedio al predecir el mejor ritmo de "
                       f"un bloque. {entrenamiento['mae']:.2f} min/km equivalen a unos "
                       f"{entrenamiento['mae'] * 60:.0f} segundos por kilómetro. Se calcula con "
                       "las mismas predicciones sin ver el bloque que usa el R².")

        mejora = 100 * (1 - entrenamiento["mae"] / entrenamiento["mae_marca_previa"])
        if mejora > 0:
            st.caption(f"Si en cada bloque se repitiera el mejor ritmo del bloque anterior, el error "
                       f"medio sería de {entrenamiento['mae_marca_previa']:.2f} min/km. El modelo lo "
                       f"deja en {entrenamiento['mae']:.2f} min/km, un {mejora:.0f} % menos.")
        else:
            st.caption(f"Repetir el mejor ritmo del bloque anterior da un error medio de "
                       f"{entrenamiento['mae_marca_previa']:.2f} min/km y el modelo no lo mejora con "
                       f"los datos actuales, que dan {entrenamiento['mae']:.2f} min/km.")

        st.divider()
        metricas = modelo.metricas_ultimo_bloque(datos_completos)

        if metricas is None:
            st.info("Para predecir hace falta al menos una carrera de 5 km o más tanto en los "
                    "últimos 28 días como en los 28 anteriores a esos. Sal a correr y vuelve a consultar.")
            prediccion = None
        else:
            col_dist, col_res = st.columns([1, 2])

            with col_dist:
                distancia = st.selectbox("Distancia objetivo", [5.0, 10.0, 15.0, 21.1],
                                         index=1, format_func=lambda d: f"{d:g} km")

            ritmo_referencia = modelo.predecir_ritmo(entrenamiento, metricas)
            tiempo = modelo.ritmo_a_tiempo(ritmo_referencia, distancia)
            tiempo_texto = modelo.formatear_tiempo(tiempo)
            # El ritmo mostrado se deriva del tiempo ya ajustado por Riegel, para que cambie
            # con la distancia elegida en lugar de mostrar siempre el mismo valor de referencia.
            ritmo_mostrado = tiempo / distancia
            escala_distancia = ritmo_mostrado / ritmo_referencia

            with col_res:
                r1, r2 = st.columns(2)
                r1.metric(f"Tiempo estimado en {distancia:g} km", tiempo_texto)
                r2.metric("Ritmo de competición",
                          f"{backend.formatear_ritmo(ritmo_mostrado)} min/km")

            if modelo.fuera_de_rango_calibrado(distancia):
                st.warning(f"⚠️ El modelo está calibrado con esfuerzos de {modelo.RANGO_CALIBRADO[0]} a "
                           f"{modelo.RANGO_CALIBRADO[1]} km. La proyección a {distancia:g} km es una extrapolación "
                           "corregida con la fórmula de Riegel y su fiabilidad es menor.")

            prediccion = {"ritmo": ritmo_mostrado, "distancia": distancia, "tiempo_texto": tiempo_texto}

            st.divider()
            st.subheader("🎯 Verificador de viabilidad")
            st.caption("Contrasta una meta contra tu progresión histórica real. No proyecta un volumen "
                       "de entrenamiento porque el historial no sostiene esa relación. Tus bloques "
                       "entrenan cantidades parecidas cada mes, y sin meses de mucho y de poco volumen "
                       "no hay contraste para medir cuánto mejora el ritmo por cada kilómetro extra. "
                       "Que no se pueda medir con estos datos no significa que el volumen no importe.")

            v1, v2 = st.columns([1, 2])

            with v1:
                st.markdown(f"**Meta en {distancia:g} km**")
                horas_defecto, minutos_defecto = divmod(int(round(tiempo * 0.95)), 60)

                col_h, col_m = st.columns(2)
                meta_horas = col_h.number_input("Horas", min_value=0, max_value=4,
                                                value=horas_defecto, step=1)
                meta_minutos = col_m.number_input("Minutos", min_value=0, max_value=59,
                                                  value=minutos_defecto, step=1)
                meta_min = meta_horas * 60 + meta_minutos

                if meta_min >= 10:
                    st.caption(f"Son {modelo.formatear_tiempo(meta_min)}, a "
                               f"{backend.formatear_ritmo(meta_min / distancia)} min/km.")

            if meta_min < 10:
                with v2:
                    st.info("Fija una meta de al menos 10 minutos para verificarla.")
            else:
                verificacion = modelo.verificar_viabilidad(entrenamiento, metricas, meta_min, distancia)

                with v2:
                    if not verificacion["viable"]:
                        st.error(verificacion["mensaje"])
                    else:
                        mejora_exigida = verificacion["mejora_exigida_min_km"]
                        diferencia_total = tiempo - meta_min

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Ritmo estimado", f"{backend.formatear_ritmo(ritmo_mostrado)} min/km",
                                  help="Es el ritmo que el modelo calcula para ti en esta distancia, "
                                       "el mismo que aparece arriba.")
                        c2.metric("Ritmo de la meta", f"{backend.formatear_ritmo(meta_min / distancia)} min/km")
                        c3.metric("Lo que exige la meta",
                                  segundos_texto(mejora_exigida) if mejora_exigida > 0 else "Nada, ya la cumples",
                                  help="Resta entre tu ritmo estimado y el ritmo de la meta, expresada "
                                       "en segundos por kilómetro. Si la meta es más lenta que tu "
                                       "estimación no exige nada.")

                        if diferencia_total > 0:
                            st.caption(f"En los {distancia:g} km son {modelo.formatear_tiempo(diferencia_total)} "
                                       "menos que tu tiempo estimado.")

                        margen = verificacion["margen_error"] * escala_distancia
                        st.caption(f"El modelo suele equivocarse unos {margen * 60:.0f} segundos por "
                                   f"kilómetro, el mismo error medio de arriba. Lo razonable es pensar en "
                                   f"un ritmo entre {backend.formatear_ritmo(ritmo_mostrado - margen)} y "
                                   f"{backend.formatear_ritmo(ritmo_mostrado + margen)} min/km.")

                        referencias = verificacion["referencias"]
                        if referencias:
                            filas_ref = [{
                                "Plazo": f"{meses} mesociclo" + ("s" if meses > 1 else "")
                                         + f", unas {meses * 4} semanas",
                                "Lo que sueles mejorar en ese plazo": segundos_texto(ref["mejora_media_min_km"]),
                                "Medido sobre": f"{ref['pares']} bloques de tu historial",
                            } for meses, ref in sorted(referencias.items())]
                            st.dataframe(pd.DataFrame(filas_ref), width="stretch", hide_index=True)

                            # Se compara contra el plazo más largo disponible, el más permisivo, porque
                            # sin un objetivo con fecha el sistema no sabe cuántas semanas quedan de verdad.
                            plazo_max = max(referencias)
                            tipica = referencias[plazo_max]["mejora_media_min_km"]

                            if mejora_exigida <= 0:
                                st.success("✅ Tu ritmo estimado ya alcanza esta meta.")
                            elif mejora_exigida <= max(tipica, 0.01):
                                st.success(f"✅ Está en línea con lo que sueles mejorar en "
                                           f"{plazo_max} mesociclos. Meta razonable con el tiempo suficiente.")
                            elif mejora_exigida <= 2 * max(tipica, 0.01):
                                st.warning(f"⚠️ Exige más del doble de lo que sueles mejorar en {plazo_max} "
                                           "mesociclos. Es alcanzable con más tiempo o revisando la meta.")
                            else:
                                st.error(f"⚠️ Muy por encima de tu progresión histórica en {plazo_max} "
                                         "mesociclos. Considera una meta más conservadora o un plazo mayor.")
                        else:
                            st.caption("Aún no hay suficiente historial consecutivo para medir tu "
                                       "progresión típica y dar una referencia.")

        st.divider()
        st.subheader("Progresión: mejor ritmo por mesociclo")

        historico = entrenamiento["datos"].reset_index().rename(columns={"index": "Mesociclo"})

        # El número de bloque no es interpretable por sí solo: se traduce a la fecha real
        # de inicio del mesociclo, contada desde la primera actividad del histórico.
        origen = datos_completos["Fecha"].min().normalize()
        historico["Inicio"] = origen + pd.to_timedelta(historico["Mesociclo"] * 28, unit="D")

        # El mejor ritmo de cada bloque oscila mucho porque depende de si ese mes hubo un
        # esfuerzo a tope. La media de tres bloques suaviza el zigzag y deja ver la tendencia.
        historico["Tendencia"] = historico["mejor_ritmo"].rolling(3, center=True, min_periods=2).mean()
        historico["Ritmo texto"] = historico["mejor_ritmo"].apply(backend.formatear_ritmo)
        historico["Tendencia texto"] = historico["Tendencia"].apply(backend.formatear_ritmo)

        fig_prog = go.Figure()
        fig_prog.add_trace(go.Scatter(
            x=historico["Inicio"], y=historico["mejor_ritmo"], mode="lines+markers",
            name="Mejor ritmo del bloque", line=dict(color="#B2BEC3", width=1.5),
            marker=dict(size=6, color="#636E72"), customdata=historico[["Ritmo texto"]],
            hovertemplate="Bloque iniciado el %{x|%d/%m/%Y}<br>"
                          "Mejor ritmo %{customdata[0]} min/km<extra></extra>"))
        fig_prog.add_trace(go.Scatter(
            x=historico["Inicio"], y=historico["Tendencia"], mode="lines",
            name="Tendencia de tres bloques", line=dict(color="#2E86DE", width=3),
            customdata=historico[["Tendencia texto"]],
            hovertemplate="Tendencia %{customdata[0]} min/km<extra></extra>"))

        minimo = int(np.floor(historico["mejor_ritmo"].min()))
        maximo = int(np.ceil(historico["mejor_ritmo"].max()))
        marcas = list(range(minimo, maximo + 1))

        # El eje se invierte porque un ritmo menor es un rendimiento mejor.
        fig_prog.update_yaxes(autorange="reversed", tickmode="array",
                              tickvals=marcas, ticktext=[str(m) for m in marcas],
                              title_text="Mejor ritmo en min/km")
        fig_prog.update_layout(height=330, margin=dict(t=10), hovermode="x unified",
                               legend=dict(orientation="h", y=1.12, x=0))
        st.plotly_chart(fig_prog, width="stretch")
        st.caption(f"Cada punto es un bloque de 28 días, contados desde la primera actividad "
                   f"{origen.strftime('%d/%m/%Y')}. Solo aparecen los bloques con al menos una carrera de 5 km o más. "
                   "El eje está invertido, cuanto más bajo sea el ritmo, más rápido. La línea azul promedia "
                   "tres bloques seguidos para que se vea hacia dónde vas sin el zigzag de cada mes.")

with tab_pulso:
    st.subheader("Pulso cargado a mano")
    st.caption("Cuando el reloj registra la frecuencia cardíaca pero no la envía a Strava, la serie se puede "
               "pegar aquí.")

    # El aviso se guarda en la sesión porque la recarga que sigue a guardar o borrar
    # descartaría de inmediato cualquier mensaje mostrado antes de ella.
    if st.session_state.get("aviso_pulso"):
        st.success(st.session_state.pop("aviso_pulso"))

    pendientes = data_loader.listar_actividades_sin_fc()
    resumen_manual = data_loader.resumen_fc_manual()
    claves_con_serie = set(resumen_manual["fecha_actividad"]) if not resumen_manual.empty else set()

    if pendientes.empty:
        st.info("Todas las actividades sincronizadas traen pulso propio de Strava. Sincroniza de nuevo "
                "si acabas de subir una salida sin frecuencia cardíaca.")
    else:
        tipos = dict(zip(pendientes["Fecha de la actividad"], pendientes["Tipo de actividad"]))
        clave = st.selectbox(
            "Actividad sin pulso en Strava",
            list(pendientes["Fecha de la actividad"]),
            format_func=lambda c: etiqueta_actividad(c, tipos.get(c, ""), c in claves_con_serie),
        )

        modelo_texto = ("Karvonen, sobre tu reserva cardíaca"
                        if ajustes.usa_karvonen(ajustes_usuario)
                        else "porcentaje de FC máxima")
        st.caption(f"Zonas por {modelo_texto}, con una FC máxima de "
                   f"**{fc_maxima_activa} ppm** {ajustes.ORIGEN_FCMAX[origen_fcmax]}. "
                   f"Se cambia en Ajustes.")

        existente = data_loader.leer_fc_manual(clave)

        col_entrada, col_vista = st.columns([1, 1])

        with col_entrada:
            # La clave lleva la actividad y un contador. Con la actividad, cada una tiene su
            # propio cuadro y el texto de una no se arrastra a la siguiente. Con el contador,
            # subirlo al guardar crea un cuadro nuevo y vacío.
            version_texto = st.session_state.get("version_texto_pulso", 0)
            texto_pegado = st.text_area(
                "Serie de tiempo y pulso",
                key=f"texto_pulso_{clave}_{version_texto}",
                height=260,
                placeholder="Tiempo_de_ruta\tFC_ppm\n00:00:00\t118\n00:00:07\t120\n00:00:26\t129",
                help="Pega el bloque completo tal como lo copias del reloj. La línea de cabecera "
                     "se descarta sola.",
            )

            serie_nueva = (data_loader.parsear_serie_fc(texto_pegado)
                           if texto_pegado.strip() else pd.DataFrame())

            if texto_pegado.strip() and serie_nueva.empty:
                st.error("No se reconoció ninguna muestra. Cada línea debe tener un tiempo "
                         "(HH:MM:SS o MM:SS) y un número de pulsaciones.")

            if st.button("Guardar serie", width="stretch", disabled=serie_nueva.empty):
                guardadas, mensaje = data_loader.guardar_fc_manual(clave, serie_nueva)
                if guardadas:
                    obtener_datos.clear()
                    st.session_state.version_datos += 1
                    st.session_state.version_texto_pulso = version_texto + 1
                    st.session_state.aviso_pulso = mensaje
                    st.rerun()
                else:
                    st.error(mensaje)

            if not existente.empty:
                st.divider()
                confirmar = st.checkbox("Confirmo que quiero descartar la serie guardada")
                if st.button("Borrar serie de esta actividad", width="stretch", disabled=not confirmar):
                    borradas, mensaje = data_loader.borrar_fc_manual(clave)
                    obtener_datos.clear()
                    st.session_state.version_datos += 1
                    if borradas:
                        st.session_state.aviso_pulso = mensaje
                        st.rerun()
                    st.error(mensaje)
                    st.rerun()

        with col_vista:
            # Se prioriza lo pegado sobre lo guardado para poder revisar antes de confirmar.
            vista_serie = serie_nueva if not serie_nueva.empty else existente
            origen_vista = "recién pegada" if not serie_nueva.empty else "guardada"

            if vista_serie.empty:
                st.info("Pega una serie a la izquierda para ver el resumen antes de guardarla.")
            else:
                tiempos = vista_serie["tiempo_s"].to_numpy(dtype=float)
                pulsos = vista_serie["fc_ppm"].to_numpy(dtype=float)
                indicadores = metricas_fc.resumen_serie(tiempos, pulsos,
                                                        fc_maxima_activa, limites_zonas)

                v1, v2, v3 = st.columns(3)
                v1.metric("Muestras", indicadores["muestras"])
                v2.metric("FC media",
                          f"{indicadores['fc_media']:.0f} ppm" if indicadores["fc_media"] else "—",
                          help="Media ponderada por tiempo: cada tramo pesa según su duración, "
                               "no según cuántas muestras contiene.")
                v3.metric("FC máxima", f"{indicadores['fc_maxima']:.0f} ppm")

                if indicadores["fc_maxima"] and indicadores["fc_maxima"] > fc_maxima_activa:
                    st.warning("Esta serie supera la referencia máxima indicada arriba. Súbela o las "
                               "zonas quedarán comprimidas hacia Z5.")

                # El eje se dibuja en minutos, pero tanto las marcas como el hover se
                # rotulan en horas y minutos: en sesiones largas «minuto 265» no se lee.
                etiquetas = [backend.formatear_duracion_corta(t) for t in tiempos]

                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=tiempos / 60, y=pulsos, mode="lines",
                    customdata=etiquetas,
                    line=dict(color="#EE5A24", width=2),
                    hovertemplate="Tiempo: %{customdata}<br>FC: %{y:.0f} ppm<extra></extra>"))

                # Las marcas se reparten en múltiplos de cinco minutos para que las
                # etiquetas caigan en valores redondos y no se solapen.
                duracion = max(float(tiempos[-1]), 1.0)
                paso = max(300, int(round(duracion / 6 / 300)) * 300)
                marcas_s = np.arange(0, duracion + 1, paso)
                fig_fc.update_xaxes(
                    tickmode="array", tickvals=marcas_s / 60,
                    ticktext=[backend.formatear_duracion_corta(s) for s in marcas_s])

                fig_fc.update_layout(height=240, margin=dict(t=10),
                                     xaxis_title="Tiempo de actividad", yaxis_title="ppm")

                # Cada franja es una zona de la configuración activa, así se ve en qué momento de
                # la sesión se pasó de una a otra. El rango vertical se fija a mano porque las
                # franjas de la Z5, que no tiene techo, ampliarían el eje sin control.
                techo = max(float(pulsos.max()), float(fc_maxima_activa)) + 6
                piso = max(30.0, float(pulsos.min()) - 8)
                for (nombre_z, desde_z, hasta_z), color_z in zip(limites_zonas, COLORES_ZONA):
                    y0, y1 = max(desde_z, piso), min(hasta_z, techo)
                    if y0 < y1:
                        fig_fc.add_hrect(y0=y0, y1=y1, fillcolor=color_z, opacity=0.28,
                                         line_width=0, layer="below",
                                         annotation_text=nombre_z, annotation_position="top left",
                                         annotation_font_size=11, annotation_font_color="#636E72")
                fig_fc.update_yaxes(range=[piso, techo])
                st.plotly_chart(fig_fc, width="stretch")

                st.caption(f"Serie {origen_vista}: {backend.formatear_duracion_corta(indicadores['duracion_s'])} "
                           "cubiertos. Los huecos entre muestras se interpolan de forma lineal.")

                st.divider()
                st.markdown("**Reparto por zonas de intensidad**")

                nombres = [nombre for nombre, _, _, _ in metricas_fc.ZONAS]
                minutos_zona = [indicadores["zonas"][n] / 60 for n in nombres]

                # El rango en pulsaciones va en la etiqueta de cada barra y el porcentaje junto
                # a los minutos, calculado sobre la duración total de la serie.
                duracion_min = max(indicadores["duracion_s"] / 60, 1e-9)
                etiquetas_zona = [f"{n}  {ajustes.texto_rango(r)}"
                                  for n, r in zip(nombres, limites_zonas)]
                textos_zona = [f"{m:.0f} min · {100 * m / duracion_min:.0f} %" for m in minutos_zona]

                fig_zonas = go.Figure(go.Bar(
                    x=minutos_zona, y=etiquetas_zona, orientation="h",
                    marker_color=COLORES_ZONA, text=textos_zona,
                    textposition="outside", cliponaxis=False,
                    hovertemplate="%{y}<br>%{x:.1f} min<extra></extra>"))
                fig_zonas.update_xaxes(range=[0, max(minutos_zona) * 1.35 + 1])
                fig_zonas.update_layout(height=240, margin=dict(t=10),
                                        xaxis_title="Minutos", yaxis_title="")
                st.plotly_chart(fig_zonas, width="stretch")

                fuera_de_zona = indicadores["duracion_s"] - sum(indicadores["zonas"].values())
                
                if fuera_de_zona > 30:
                    texto_fuera = (f" Además, {fuera_de_zona / 60:.0f} minutos por debajo del inicio de la "
                               "Z1 no entran en ninguna zona." if fuera_de_zona > 30 else "")
                    st.caption("Los porcentajes se calculan sobre la duración total de la serie." + texto_fuera)
                
                z1, z2 = st.columns(2)
                z1.metric("TRIMP de Edwards", f"{indicadores['trimp_edwards']:.0f}",
                          help="Suma de los minutos de cada zona por su multiplicador de intensidad "
                               "(1 a 5). Es una métrica complementaria: la columna Carga que alimenta "
                               "el ACWR sigue usando la fórmula original para no mezclar escalas.")

                deriva = indicadores["deriva"]
                if deriva["deriva_pct"] is None:
                    z2.metric("Deriva cardíaca", "—")
                    st.caption("La serie es demasiado corta para comparar las dos mitades.")
                else:
                    z2.metric("Deriva cardíaca", f"{deriva['deriva_pct']:+.1f} %",
                              help="Diferencia del pulso medio entre la segunda mitad y la primera. "
                                   "Un valor positivo con el mismo esfuerzo indica fatiga o "
                                   "deshidratación.")
                    if not deriva["estable"]:
                        st.caption(f"El pulso varió un {deriva['variabilidad_pct']:.0f} % a lo largo de la "
                                   "sesión, así que no fue un esfuerzo continuo. En series o cuestas "
                                   "repetidas la deriva no es interpretable.")

    if not resumen_manual.empty:
        st.divider()
        st.subheader("Series ya cargadas")

        tabla = resumen_manual.copy()
        tabla["Actividad"] = tabla["fecha_actividad"].apply(
            lambda c: etiqueta_actividad(c, "", False).replace(" — ", ""))
        tabla["FC media"] = tabla["fc_media_manual"].round(0)
        tabla["FC máxima"] = tabla["fc_maxima_manual"].round(0)
        st.dataframe(tabla[["Actividad", "FC media", "FC máxima"]],
                     width="stretch", hide_index=True)

with tab_hist:
    st.subheader("Resumen por deporte")

    resumen = backend.resumen_por_tipo(datos_completos)
    if not resumen.empty:
        st.dataframe(resumen, width="stretch", hide_index=True)

    st.divider()
    st.subheader(f"Distribución de distancias — {deporte} ({año})")

    fig_hist = px.histogram(datos, x="Distancia_km", nbins=25,
                            labels={"Distancia_km": "Distancia (km)"},
                            color_discrete_sequence=["#EE5A24"])
    fig_hist.update_traces(hovertemplate="Distancia: %{x:.1f} km<br>No. actividades: %{y}<extra></extra>")
    fig_hist.update_layout(height=300, margin=dict(t=10), yaxis_title="No. actividades",
                           xaxis_tickformat=".1f", bargap=0.05)
    st.plotly_chart(fig_hist, width="stretch")
    st.caption(f"Distancias de tus {len(datos)} actividades de {deporte.lower()} en el periodo seleccionado. "
               "Sirve para ver a qué distancias entrenas habitualmente.")

    st.divider()
    st.subheader("Últimas actividades")
    st.dataframe(backend.ultimas_actividades(datos, 10, limites_zonas),
                 width="stretch", hide_index=True)
    st.caption("«Tiempo en movimiento» alimenta el cálculo de carga. «Tiempo total» "
               "incluye las paradas y es el que cuenta como marca oficial en competición.")

with tab_agente:
    st.subheader("🤖 Coach AI")

    if "mensajes" not in st.session_state:
        st.session_state.mensajes = agente.cargar_historial()

    cabecera, boton = st.columns([4, 1])
    cabecera.caption("El agente recibe tus indicadores, la predicción del modelo y tu diario de estado.")

    if boton.button("🗑️ Borrar memoria", width="stretch"):
        agente.borrar_historial()
        st.session_state.mensajes = []
        st.session_state.pop("huella_actividades", None)
        st.rerun()

    # La conversación se confina a un contenedor de altura fija con desplazamiento propio,
    # para que el historial no alargue la página. Por defecto solo se muestran los últimos
    # intercambios, de modo que la pregunta más reciente quede siempre a la vista.
    MENSAJES_VISIBLES = 6
    total_mensajes = len(st.session_state.mensajes)

    ver_todo = st.checkbox(f"Ver conversación completa ({total_mensajes} mensajes)",
                           value=False, disabled=total_mensajes <= MENSAJES_VISIBLES)

    visibles = st.session_state.mensajes if ver_todo else st.session_state.mensajes[-MENSAJES_VISIBLES:]

    ventana_chat = st.container(height=420)
    with ventana_chat:
        for mensaje in visibles:
            with st.chat_message(mensaje["role"]):
                st.markdown(mensaje["content"])

    entrada_chat = st.container()

    if prompt := entrada_chat.chat_input(
            "Ej: ¿Qué entreno mañana? ¿Voy bien para bajar de 50 min en 10K?"):

        st.session_state.mensajes.append({"role": "user", "content": prompt})
        with ventana_chat:
            with st.chat_message("user"):
                st.markdown(prompt)

        if not API_KEY_CLAUDE:
            st.error("Falta 'ANTHROPIC_API_KEY'. Añádela al .env en local o a los Secrets de Streamlit Cloud.")
        else:
            try:
                cliente = anthropic.Anthropic(api_key=API_KEY_CLAUDE)

                entrenamiento_ctx = modelo.entrenar_modelo(datos_completos)
                metricas_ctx = modelo.metricas_ultimo_bloque(datos_completos)
                prediccion_ctx = None

                if entrenamiento_ctx and metricas_ctx:
                    ritmo_ctx = modelo.predecir_ritmo(entrenamiento_ctx, metricas_ctx)
                    prediccion_ctx = {
                        "ritmo": ritmo_ctx,
                        "distancia": 10.0,
                        "tiempo_texto": modelo.formatear_tiempo(modelo.ritmo_a_tiempo(ritmo_ctx, 10.0)),
                    }

                recientes = backend.ultimas_actividades(
                    datos_completos, ajustes_usuario["actividades_agente"],
                    limites_zonas, tiempo_legible=False)

                # Se compara con lo que vio el modelo en el mensaje anterior para decirle
                # explícitamente qué cambió, en vez de esperar que lo note por su cuenta.
                huella = agente.huella_actividades(recientes)
                cambios = agente.cambios_de_pulso(
                    st.session_state.get("huella_actividades", {}), huella)
                st.session_state.huella_actividades = huella

                contexto = agente.construir_contexto(
                    kpis_hoy, diagnostico, prediccion_ctx, entrenamiento_ctx, agente.cargar_diario(),
                    actividades_recientes=recientes, cambios=cambios,
                )

                st.session_state.ultimo_contexto = contexto
                aviso = None
                if cambios:
                    aviso = ("Desde tu respuesta anterior el atleta actualizó estos datos, que ya "
                            "están en tu contexto: " + " ".join(cambios))

                with ventana_chat:
                    with st.chat_message("assistant"):
                        with st.spinner("Analizando tus datos..."):
                            texto = agente.consultar_agente(cliente, contexto,
                                                            st.session_state.mensajes, aviso)
                        st.markdown(texto)

                st.session_state.mensajes.append({"role": "assistant", "content": texto})
                agente.guardar_historial(st.session_state.mensajes)

            except Exception as error:
                st.error(f"No se pudo consultar al agente: {error}")

            # Panel de diagnóstico, oculto de la interfaz por ahora. Descomentar si hace falta
            # depurar por qué el agente respondió algo raro.
            # with st.expander("🔍 Ver lo que recibió el agente en el último mensaje"):
            #     st.code(st.session_state.get("ultimo_contexto", "Todavía no hay mensajes."),
            #             language=None)
                
with tab_ajustes:
    st.subheader("Ajustes personales")
    st.caption("Se guardan en la base de datos, así que siguen puestos la próxima vez "
               "que abras la aplicación.")

    if st.session_state.pop("ajustes_guardados", False):
        st.success("Ajustes guardados.")

    actuales = ajustes_usuario
    col_perfil, col_fc = st.columns(2)

    with col_perfil:
        st.markdown("**Perfil**")
        nacimiento = st.date_input(
            "Fecha de nacimiento", value=actuales["fecha_nacimiento"],
            min_value=date(1940, 1, 1), max_value=config.hoy(), format="DD/MM/YYYY",
            help="De aquí sale la edad y, con ella, la estimación de Tanaka.")

        edad_actual = ajustes.edad({"fecha_nacimiento": nacimiento})
        if edad_actual is None:
            st.caption("Sin fecha de nacimiento no se puede estimar la FC máxima por edad.")
        else:
            tanaka = ajustes.fc_maxima_tanaka({"fecha_nacimiento": nacimiento})
            st.caption(f"Edad: {edad_actual} años. Tanaka estimaría {tanaka} ppm.")

    with col_fc:
        st.markdown("**Frecuencia cardíaca**")

        medida_guardada = actuales["fc_maxima_medida"]
        origen_fcmax = st.radio(
            "De dónde sale la FC máxima", ["Valor medido", "Estimación por edad"],
            index=0 if medida_guardada else 1, horizontal=True,
            help="Tanaka se desvía varios latidos de una persona a otra, así que un valor "
                 "medido de verdad siempre manda sobre la fórmula.")

        # El campo numérico nunca queda vacío porque Streamlit repone el último valor
        # válido al borrarlo, así que quién manda lo decide el selector de arriba.
        valor_medido = st.number_input(
            "FC máxima medida (ppm)", min_value=140, max_value=230,
            value=medida_guardada or config.FC_MAXIMA, step=1,
            disabled=(origen_fcmax == "Estimación por edad"),
            help="El valor más alto que hayas alcanzado de verdad y hayas comprobado.")

        fc_maxima_medida = valor_medido if origen_fcmax == "Valor medido" else None

        fc_reposo = st.number_input(
            "FC en reposo (ppm)", min_value=30, max_value=100,
            value=actuales["fc_reposo"], step=1,
            help="Mídela acostado antes de levantarte, varias mañanas, y usa el valor más "
                 "bajo de una semana normal. Hace falta para Karvonen.")

    activa, origen = ajustes.fc_maxima_efectiva(
        {"fc_maxima_medida": fc_maxima_medida, "fecha_nacimiento": nacimiento})
    st.info(f"Athletix usará **{activa} ppm** como FC máxima, {ajustes.ORIGEN_FCMAX[origen]}.")

    modelo_zonas = st.radio(
        "Modelo de zonas", ["fcmax", "karvonen"],
        index=0 if actuales["modelo_zonas"] == "fcmax" else 1, horizontal=True,
        format_func=lambda v: ("Porcentaje de FC máxima" if v == "fcmax"
                               else "Karvonen, por reserva cardíaca"))

    if modelo_zonas == "karvonen" and not fc_reposo:
        st.warning("Karvonen necesita la FC en reposo. Mientras esté vacía se seguirá "
                   "usando el porcentaje de FC máxima.")

    previa = {"fc_maxima_medida": fc_maxima_medida, "fecha_nacimiento": nacimiento,
              "fc_reposo": fc_reposo}
    por_maxima = ajustes.rangos_de_zonas({**previa, "modelo_zonas": "fcmax"})
    por_karvonen = ajustes.rangos_de_zonas({**previa, "modelo_zonas": "karvonen"})

    karvonen_activo = modelo_zonas == "karvonen" and bool(fc_reposo)
    titulo_maxima = "% de FC máxima" + ("" if karvonen_activo else "  ← en uso")
    titulo_karvonen = "Karvonen" + ("  ← en uso" if karvonen_activo else "")

    comparativa = pd.DataFrame({
        "Zona": [nombre for nombre, _, _ in por_maxima],
        titulo_maxima: [ajustes.texto_rango(r) for r in por_maxima],
        titulo_karvonen: ([ajustes.texto_rango(r) for r in por_karvonen] if fc_reposo
                          else ["—"] * len(por_maxima)),
    })
    st.dataframe(comparativa, width="stretch", hide_index=True)
    st.caption("Las dos columnas se muestran siempre para que se puedan comparar. El "
               "selector de arriba no cambia la tabla, marca cuál de las dos aplica "
               "Athletix.")

    st.divider()
    col_barra, col_agente = st.columns(2)

    with col_barra:
        st.markdown("**Arranque de la barra lateral**")
        deporte_defecto = st.selectbox(
            "Deporte al abrir", deportes,
            index=ajustes.indice_por_defecto(actuales, "deporte_defecto", deportes))

        opciones_periodo = ["Año actual", "Todo el histórico"] + [str(a) for a in años]
        periodo_defecto = st.selectbox(
            "Periodo al abrir", opciones_periodo,
            index=(opciones_periodo.index(actuales["periodo_defecto"])
                   if actuales["periodo_defecto"] in opciones_periodo else 0),
            help="'Año actual' se recalcula solo cada enero. Un año concreto se queda fijo.")

    with col_agente:
        st.markdown("**Agente y diario**")
        actividades_agente = st.slider(
            "Actividades que recibe el agente", min_value=5, max_value=50,
            value=int(actuales["actividades_agente"]), step=5,
            help="Más contexto sube el costo y la latencia de cada respuesta.")

        retencion_diario = st.number_input(
            "Registros del diario que se conservan", min_value=7, max_value=365,
            value=int(actuales["retencion_diario"]), step=1,
            help="Son registros, no días naturales: si anotas tres veces por semana, "
                 "treinta registros cubren unas diez semanas.")

    st.divider()

    if st.button("Guardar ajustes", type="primary", width="stretch"):
        if ajustes.guardar({
            "fecha_nacimiento": nacimiento,
            "fc_maxima_medida": int(fc_maxima_medida) if fc_maxima_medida else None,
            "fc_reposo": int(fc_reposo) if fc_reposo else None,
            "modelo_zonas": modelo_zonas,
            "deporte_defecto": deporte_defecto,
            "periodo_defecto": periodo_defecto,
            "actividades_agente": int(actividades_agente),
            "retencion_diario": int(retencion_diario),
        }):
            # Se recarga la página entera para que la barra lateral y las demás pestañas
            # recojan los valores nuevos ya mismo, y no en la siguiente interacción.
            st.session_state.ajustes_guardados = True
            st.rerun()
        else:
            st.error("No se pudieron guardar. Revisa la conexión con la base de datos.")

# Streamlit no ofrece un modo oficial de desplazar la vista al abrir una pestaña, así que
# se inyecta un script mínimo. No se busca la pestaña, cuya estructura interna cambia
# entre versiones, sino el momento en que la caja de escritura pasa a estar visible.
SCRIPT_IR_AL_CHAT = """
<script>
const padre = window.parent;
const doc = padre.document;

function cajaDeEscritura() {
    return doc.querySelector('textarea[placeholder^="Ej:"]')
        || doc.querySelector('[data-testid="stChatInput"]');
}

function irAlChat(entrada) {
    // Se sube la última pregunta y no el final de la respuesta, porque una respuesta
    // larga obligaría a leerla empezando por su última línea.
    const mensajes = doc.querySelectorAll('[data-testid="stChatMessage"]');
    const objetivo = mensajes[Math.max(mensajes.length - 2, 0)];
    if (objetivo) objetivo.scrollIntoView({block: "start"});
    entrada.scrollIntoView({block: "end"});
}

let visibleAntes = false;
if (padre.__athletixVigilante) padre.clearInterval(padre.__athletixVigilante);
padre.__athletixVigilante = padre.setInterval(() => {
    const entrada = cajaDeEscritura();
    const visible = !!(entrada && entrada.getBoundingClientRect().height > 0);
    if (visible && !visibleAntes) irAlChat(entrada);
    visibleAntes = visible;
}, 300);

// Mensajes: __TOTAL__. El número cambia con cada mensaje nuevo y obliga a reiniciar el
// vigilante, que entonces vuelve a bajar la vista hasta la respuesta recién llegada.
</script>
"""

st.iframe(SCRIPT_IR_AL_CHAT.replace("__TOTAL__", str(len(st.session_state.mensajes))),
          height=1)