"""Sincroniza el histórico completo de Strava con Supabase desde la consola.

Hace lo mismo que el botón Sincronizar con Strava de la aplicación. Se ejecuta a mano
desde la raíz del proyecto y no hace falta vaciar la tabla antes, porque cada
sincronización revisa el historial entero.
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
    cuantas, mensaje = data_loader.sincronizar_con_strava()
    print(mensaje)