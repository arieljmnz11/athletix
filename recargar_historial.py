"""Descarga de nuevo el histórico completo de Strava y lo guarda en Supabase.

Operación de mantenimiento puntual: se ejecuta a mano desde la raíz del proyecto,
después de vaciar la tabla, y no forma parte de la aplicación.
"""

import os
import data_loader

CLAVES = ["SUPABASE_URL", "SUPABASE_KEY", "STRAVA_CLIENT_ID",
          "STRAVA_CLIENT_SECRET", "STRAVA_REFRESH_TOKEN"]

faltan = [clave for clave in CLAVES if not os.getenv(clave)]

if faltan:
    print("Faltan estas credenciales en tu archivo .env local:")
    for clave in faltan:
        print(f"   - {clave}")
    print("Cópialas de los Secrets de Streamlit Cloud y vuelve a ejecutar.")
else:
    print("Descargando todo el histórico desde Strava. Tarda alrededor de un minuto.")
    cuantas, mensaje = data_loader.sincronizar_con_strava(None)
    print(mensaje)