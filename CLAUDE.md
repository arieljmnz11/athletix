# Athletix

Aplicación de Streamlit que analiza el historial deportivo de Strava de Ariel, running y
ciclismo. Calcula carga de entrenamiento y ACWR, estima rendimiento con una regresión por
mesociclos, y expone un agente conversacional que interpreta esos indicadores.

## Cómo trabajar conmigo

Soy Ariel, estudiante de Ciencia de Datos e IA en la UNACH. Athletix es un pasatiempo y
soy aprendiz de Python. No leo las líneas de código ni conozco el flujo completo del
programa, así que:

- Un punto a la vez, en orden de dependencias, sin saltar etapas.
- Explica en lenguaje llano qué significa cada término técnico antes de usarlo.
- Dime exactamente qué debo probar y qué debería ver en pantalla si funcionó.
- Avísame cuándo algo es solo para probar y cuándo es versión final.
- Si me dices Ctrl+C, aclara que eso solo aplica si el proceso ya está corriendo.
- Lee el código real antes de proponer un cambio, no asumas cómo está.
- Si detectas bugs que no te señalé, dímelos.
- Ten cuidado al tocar partes que ya funcionaban.
- Al terminar cada cambio, dame un Summary y una Description cortos para el commit.
- Escribo en español informal.

Cuando algo no se pueda medir con los datos disponibles, dímelo en vez de forzar un
número. Prefiero un resultado honesto a uno vistoso.

## Estilo de código

**Comentarios internos.** Explican el porqué, no el qué. El código debe ser
autodocumentado y los comentarios solo aclaran decisiones de diseño, lógica no trivial o
restricciones. Tercera persona. Máximo una o dos líneas, inmediatamente arriba del bloque
relevante. Sin separadores decorativos.

**Docstrings.** Funciones simples, una línea en imperativo. Funciones con parámetros,
resumen directo seguido de `Args:` y `Returns:`.

**Textos de la interfaz.** Evita los dos puntos y los paréntesis. Yo ajusto el tono de
varios mensajes por mi cuenta, así que si reemplazas un bloque que ya tiene texto mío,
respeta mi redacción y no la reviertas.

**Nada de desplegables explicativos en la interfaz.** Las explicaciones largas van en el
tooltip de ayuda de un control, en `comparar_modelos.py` o en el README. Si algo deja de
mostrarse, se comenta con `#` en vez de borrarlo.

## Arquitectura

| Módulo | Responsabilidad |
|---|---|
| `app.py` | Interfaz Streamlit, siete pestañas incluidas Ajustes y Coach AI |
| `backend.py` | Limpieza, carga de entrenamiento, ACWR, agregaciones, formateadores |
| `modelo.py` | Regresión por mesociclos, validación LOO, Riegel, verificador de viabilidad |
| `agente.py` | Contexto del LLM, detección de cambios entre mensajes, persistencia |
| `data_loader.py` | API de Strava, Supabase, pulso manual, lectura y escritura de ajustes |
| `metricas_fc.py` | Funciones puras de series de pulso, zonas por porcentaje o Karvonen |
| `ajustes.py` | Lectura, escritura y valores derivados de los ajustes personales |
| `config.py` | Zona horaria y utilidades de fecha |
| `comparar_modelos.py` | Script de consola, documenta y recalcula por qué se eligió el modelo |
| `recargar_historial.py` | Mantenimiento puntual, recarga el histórico completo desde Strava |

`metricas_fc.py` existe aparte porque lo necesitan `data_loader` y `backend`, y alojarlo
en cualquiera de los dos crearía un import circular.

## Persistencia

PostgreSQL en Supabase. El disco de Streamlit Community Cloud es efímero, nada se guarda
en archivos. Cinco tablas:

- `actividades_sincronizadas`, espejo de Strava, unas 1377 filas desde el 09/10/2020.
  Clave única `fecha_actividad`, de tipo texto. Incluye `fc_maxima` y `strava_id`.
- `fc_manual`, una fila por muestra de pulso, única por `fecha_actividad` y `tiempo_s`.
  Siempre se lee ordenada por tiempo dentro de cada actividad, nunca por id de inserción.
- `historial_chat`, memoria del agente.
- `diario_estado`, estado físico diario, `fecha` como clave primaria.
- `ajustes`, fila única con `check (id = 1)`.

El `.env` local tiene las credenciales, así que cualquier script que ejecutes en esta
carpeta lee y escribe en el Supabase real. Ten cuidado con lo que escribes.

Los cambios de esquema los hago yo a mano en el editor de Supabase. Si hace falta uno,
dame el SQL y lo corro, no intentes aplicarlo desde aquí.

## Cosas que ya decidí, no hace falta rediscutirlas

- El CSV del repositorio aporta cero filas mientras Supabase esté vivo, porque la API
  cubre todo el historial y una regla de corte por `Origen` le deja al CSV solo lo
  anterior a su primer registro. El CSV se mantiene como copia de seguridad.
- El TRIMP de Edwards no alimenta la columna `Carga`, para no mezclar escalas en el ACWR,
  y se calcula siempre sobre porcentaje de FC máxima, nunca sobre Karvonen.
- La carga usa tiempo de movimiento. El tiempo transcurrido se guarda aparte porque es el
  que cuenta como marca oficial en competición.
- La media del pulso manual es por regla del trapecio, no aritmética, porque el muestreo
  es irregular.
- La fecha se ancla a `America/Guayaquil` porque el servidor de Streamlit corre en UTC.
  Usa `config.hoy()`, nunca la fecha del sistema.
- Mi FC máxima medida es 199 ppm y mi FC en reposo ronda los 50. El modelo de zonas activo
  es porcentaje de FC máxima. Karvonen está configurado pero sin activar.
- Zona horaria y peso corporal quedan fuera de Ajustes a propósito.
- Los umbrales del ACWR, 0.8, 1.3 y 1.5, quedan fijos porque vienen de literatura.
- El modelo de predicción se redujo a una sola variable, el mejor ritmo del mesociclo
  anterior, tras comparar seis variantes con validación Leave-One-Out contra dos
  referencias ingenuas. El planificador que despejaba volumen semanal se convirtió en un
  verificador de viabilidad porque el historial no tiene variación suficiente de volumen
  entre bloques. Un factor de desnivel para incluir la montaña se descartó por
  inconsistente entre cuatro métodos. Todo esto se recalcula con `comparar_modelos.py`.

## Al tocar el código

- Si un cambio toca un módulo distinto de `app.py`, hay que usar Reboot app en el panel de
  Streamlit. El proceso reutiliza los módulos ya importados.
- En local, para que un cambio fuera de `app.py` surta efecto, hay que detener la app con
  Ctrl+C en su terminal y volver a lanzarla con `streamlit run app.py`.
- `comparar_modelos.py` se puede ejecutar cuando haga falta, no forma parte de la app.
- Al cambiar el modelo o los filtros de actividades, ejecútalo y compara los números con
  la ejecución anterior.

## Qué viene después

El trabajo grande está repartido en etapas, una por vez:

1. **Objetivos.** Tabla de objetivos de carrera con fecha, distancia, desnivel y tiempo
   meta. Disciplina principal, fuerza y disponibilidad en Ajustes. Que el agente y el
   predictor usen el objetivo activo.
2. **Series de Strava.** Traer las series de pulso, pendiente y altitud bajo demanda desde
   la API, sin guardarlas. De ahí sale el Grade Adjusted Pace real, que es lo que falta
   para que el modelo entienda la montaña, más la estimación de potencia en ciclismo.
3. **Planificador con calendario.** La IA propone una semana, el código la valida contra
   el ACWR y la regla del 10 % antes de mostrarla.
4. **Limpieza.** Modularizar `app.py` en una carpeta `vistas/` con un archivo por pestaña,
   y migrar los comentarios de todos los módulos al estilo de arriba. Las dos cosas tocan
   todos los archivos, así que se hacen juntas.
