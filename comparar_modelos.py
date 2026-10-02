"""Informe del modelo de predicción de Athletix.

Se ejecuta a mano desde la raíz del proyecto con `python comparar_modelos.py`. Lee el
historial real, repite todas las comparaciones y escribe en consola un informe con los
números del momento. No modifica ningún dato ni forma parte de la aplicación.

Sustituye al desplegable explicativo que había en la pestaña Predicción y Plan. Conviene
volver a ejecutarlo cada cierto tiempo, porque cada actividad nueva puede cambiar las
conclusiones, en especial la del desnivel, que depende de cuántas salidas de montaña
distintas haya en el historial.
"""

import textwrap

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

import backend
import modelo

ANCHO = 92
# Rango ancho a propósito. El modelo en producción descarta ritmos por encima de 10 min/km
# y con ello toda la montaña, mientras que este informe necesita verla para evaluarla.
RANGO_RITMO_BRUTO = (3.0, 25.0)
DISTANCIA_MINIMA_EQ = 5.0
PENDIENTE_MINIMA = 20.0      # Metros de subida por kilómetro para considerar que hay montaña
DISPERSION_ACEPTABLE = 1.5   # Cociente entre la mayor y la menor estimación del desnivel


def parrafo(texto):
    """Imprime un párrafo ajustado al ancho del informe."""
    print(textwrap.fill(texto, ANCHO))
    print()


def seccion(numero, titulo):
    """Imprime el encabezado de una sección."""
    print("\n" + "=" * ANCHO)
    print(f"{numero}. {titulo}")
    print("=" * ANCHO + "\n")


def segundos(min_km):
    """Convierte minutos por kilómetro en segundos por kilómetro redondeados."""
    return f"{min_km * 60:.0f} s/km"


# ----------------------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------------------

def carreras_validas(df):
    """Selecciona carreras con distancia y duración creíbles, montaña incluida."""
    ritmo_bruto = df["Minutos"] / df["Distancia_km"]
    validas = df[(df["Tipo de actividad"] == "Carrera")
                 & (df["Distancia_km"] >= 3.0)
                 & ritmo_bruto.between(*RANGO_RITMO_BRUTO)].copy()
    validas["Desnivel positivo"] = validas["Desnivel positivo"].fillna(0)
    validas["ritmo_llano"] = validas["Minutos"] / validas["Distancia_km"]
    validas["pendiente"] = validas["Desnivel positivo"] / validas["Distancia_km"]
    validas["metros_por_hora"] = validas["Desnivel positivo"] / (validas["Minutos"] / 60)
    return validas


def _metros_por_km(min_por_km, min_por_metro):
    """Traduce los dos costes a metros de subida equivalentes a un kilómetro llano."""
    return min_por_km / min_por_metro if min_por_metro > 1e-9 else None


def calibrar_desnivel(carreras):
    """Estima el coste del desnivel por cuatro caminos distintos.

    Returns:
        dict: Nombre del método y la cifra de metros de subida que equivalen a un
            kilómetro llano, o None si el método no produce un coste positivo.
    """
    resultados = {}
    X = carreras[["Distancia_km", "Desnivel positivo"]].to_numpy()
    y = carreras["Minutos"].to_numpy()

    sin_corte = LinearRegression(fit_intercept=False).fit(X, y)
    resultados["1. Tiempo total, sin término independiente"] = _metros_por_km(*sin_corte.coef_)

    con_corte = LinearRegression().fit(X, y)
    resultados["2. Tiempo total, con término independiente"] = _metros_por_km(*con_corte.coef_)

    g = carreras[["pendiente"]].to_numpy()
    ritmo = LinearRegression().fit(g, carreras["ritmo_llano"].to_numpy())
    resultados["3. Ritmo contra pendiente media"] = _metros_por_km(ritmo.intercept_, ritmo.coef_[0])

    montana = carreras[carreras["pendiente"] >= PENDIENTE_MINIMA]
    if len(montana) >= 10:
        g_m = montana[["pendiente"]].to_numpy()
        ritmo_m = LinearRegression().fit(g_m, montana["ritmo_llano"].to_numpy())
        resultados[f"4. Ritmo contra pendiente, solo las {len(montana)} carreras con subida"] = \
            _metros_por_km(ritmo_m.intercept_, ritmo_m.coef_[0])
    return resultados


def construir_dataset(df, metros_por_km):
    """Arma un registro por mesociclo con todas las variables candidatas."""
    carreras = carreras_validas(df)
    carreras["km_eq"] = carreras["Distancia_km"] + carreras["Desnivel positivo"] / metros_por_km
    carreras["ritmo_eq"] = carreras["Minutos"] / carreras["km_eq"]

    largas = carreras[carreras["km_eq"] >= DISTANCIA_MINIMA_EQ]
    resistencia = df[df["Tipo de actividad"].isin(["Carrera", "Bicicleta"])]

    datos = pd.DataFrame(index=sorted(largas["Mesociclo"].unique()))
    datos["ritmo_medio_eq"] = largas.groupby("Mesociclo")["ritmo_eq"].mean()
    datos["km_semana"] = largas.groupby("Mesociclo")["Distancia_km"].sum() / 4
    datos["km_eq_semana"] = largas.groupby("Mesociclo")["km_eq"].sum() / 4
    datos["desnivel_semana"] = largas.groupby("Mesociclo")["Desnivel positivo"].sum() / 4
    datos["horas_semana"] = resistencia.groupby("Mesociclo")["Minutos"].sum() / 4 / 60
    datos["mejor_ritmo"] = largas.groupby("Mesociclo")["ritmo_eq"].min()
    datos[["horas_semana", "desnivel_semana"]] = datos[["horas_semana", "desnivel_semana"]].fillna(0)

    # El mejor ritmo del bloque anterior solo vale si los bloques son consecutivos. Un
    # salto en el índice significa meses sin correr y rompería la continuidad.
    datos["mejor_ritmo_previo"] = datos["mejor_ritmo"].shift(1)
    datos.loc[datos.index.to_series().diff() != 1, "mejor_ritmo_previo"] = np.nan
    return datos


def evaluar(datos, caracteristicas):
    """Valida con Leave-One-Out una combinación de variables y devuelve R² y MAE."""
    X, y = datos[caracteristicas], datos["mejor_ritmo"]
    pred = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
    return r2_score(y, pred), mean_absolute_error(y, pred)


# ----------------------------------------------------------------------------------------
# Secciones del informe
# ----------------------------------------------------------------------------------------

def introduccion():
    """Explica los términos que se usan en el resto del informe."""
    print("INFORME DEL MODELO DE PREDICCIÓN DE ATHLETIX")
    print("-" * ANCHO + "\n")
    parrafo("El modelo intenta predecir el mejor ritmo que se alcanzará en un bloque de 28 "
            "días, llamado mesociclo, a partir de lo ocurrido antes. Este informe repite las "
            "comparaciones que llevaron a elegir el modelo actual y las vuelve a calcular con "
            "el historial de hoy, así que las cifras cambian cuando entran actividades nuevas.")
    parrafo("MAE significa error absoluto medio. Es cuánto se equivoca el modelo en promedio, "
            "medido en minutos por kilómetro. Un MAE de 0.36 min/km equivale a unos 22 "
            "segundos por kilómetro. Cuanto más bajo, mejor.")
    parrafo("R² mide qué parte de la variación entre bloques logra explicar el modelo. Vale 1 "
            "si acierta siempre. Con 0 acierta lo mismo que repetir siempre el ritmo medio "
            "histórico, y por debajo de 0 lo hace peor que eso.")
    parrafo("Leave-One-Out es la forma de calcular ambos sin hacer trampa. Se aparta un bloque, "
            "se entrena el modelo con todos los demás, se predice el bloque apartado y se "
            "repite con cada bloque. Así cada predicción se hace sobre un bloque que el modelo "
            "nunca vio.")
    parrafo("Todo modelo se compara contra dos referencias ingenuas. La primera es repetir "
            "siempre el ritmo medio de todo el historial. La segunda, más exigente, es repetir "
            "el mejor ritmo del bloque anterior tal cual. Un modelo que no gane la segunda no "
            "aporta nada, porque bastaría con decir que este mes irás como el pasado.")


def seccion_modelo_original(df):
    """Explica por qué se descartó el modelo de tres variables."""
    seccion(1, "EL MODELO ORIGINAL Y POR QUÉ SE DESCARTÓ")
    parrafo("El modelo original usaba tres variables para predecir el mejor ritmo de un bloque. "
            "La primera era el ritmo medio de las carreras de ese mismo bloque. La segunda eran "
            "los kilómetros de carrera por semana. La tercera eran las horas semanales de "
            "carrera y bicicleta juntas.")
    parrafo("El problema está en la primera. Predecir el valor más bajo de un conjunto de "
            "ritmos usando el promedio de ese mismo conjunto es aritmética, porque el mínimo de "
            "un grupo de números siempre anda cerca de su promedio. Esa variable daba un R² "
            "alto que no decía nada sobre entrenamiento. Al medirlo de forma honesta, contra "
            "las referencias y con Leave-One-Out sobre los mismos bloques, ese R² desaparece. "
            "La tabla de la sección 3 muestra el resultado actual de esa variante, llamada A.")


def seccion_desnivel(df):
    """Calibra el coste del desnivel y devuelve el factor usado por las variantes."""
    seccion(2, "CALIBRACIÓN DEL DESNIVEL, PARA INCLUIR LAS CARRERAS DE MONTAÑA")
    carreras = carreras_validas(df)
    parrafo("Una carrera de montaña sale con un ritmo lentísimo si solo se mira la distancia, "
            "y por eso el modelo la descarta. La idea fue convertir el desnivel en kilómetros "
            "llanos equivalentes, o sea decir que tantos metros de subida cuestan lo mismo "
            "que un kilómetro plano. Ese número se puede estimar con tus propias carreras, "
            "pero hay varias maneras de hacerlo y conviene probarlas todas.")
    parrafo(f"Se analizan {len(carreras)} carreras, de las cuales "
            f"{(carreras['pendiente'] >= PENDIENTE_MINIMA).sum()} tienen {PENDIENTE_MINIMA:.0f} "
            "metros de subida por kilómetro o más. Los cuatro métodos son estos. El primero "
            "ajusta el tiempo total de cada carrera contra sus kilómetros y su desnivel, "
            "obligando a que una carrera de cero kilómetros dure cero. El segundo hace lo mismo "
            "pero deja libre ese punto de partida. El tercero ajusta el ritmo de cada carrera "
            "contra su pendiente media, de modo que cada salida pese igual sin importar su "
            "duración. El cuarto repite el tercero usando solo las carreras con subida.")

    calibraciones = calibrar_desnivel(carreras)
    print(f"  {'Método':<68}{'Metros de subida = 1 km llano':>20}")
    print("  " + "-" * 88)
    for nombre, metros in calibraciones.items():
        print(f"  {nombre:<68}{(f'{metros:.0f}' if metros else 'no calculable'):>20}")
    print()

    validas = [m for m in calibraciones.values() if m]
    factor = calibraciones["3. Ritmo contra pendiente media"] or float(np.median(validas))
    if len(validas) >= 2:
        razon = max(validas) / min(validas)
        parrafo(f"La estimación más alta es {razon:.1f} veces la más baja.")
        if razon > DISPERSION_ACEPTABLE:
            parrafo("Cuando cuatro maneras de medir lo mismo se separan tanto, la conclusión no "
                    "es elegir la que más convenga. Es que los datos no contienen una respuesta "
                    "fiable, así que el factor de desnivel no se usa en el modelo en producción.")
        else:
            parrafo("Las estimaciones coinciden razonablemente. Con este historial el factor de "
                    "desnivel empieza a ser de fiar y vale la pena reconsiderar incluirlo.")

    top = carreras.nlargest(8, "Desnivel positivo")
    llano = carreras.loc[carreras["pendiente"] < 10, "ritmo_llano"].median()
    print("  Las ocho carreras con más desnivel, y lo que habría predicho el factor del método 3")
    print(f"  {'Fecha':<12}{'km':>7}{'D+ m':>8}{'Real min':>10}{'Estimado':>10}{'Error':>8}{'m/h de subida':>15}")
    for _, f in top.iterrows():
        estimado = (f["Distancia_km"] + f["Desnivel positivo"] / factor) * llano
        print(f"  {f['Fecha'].strftime('%d/%m/%Y'):<12}{f['Distancia_km']:>7.1f}"
              f"{f['Desnivel positivo']:>8.0f}{f['Minutos']:>10.0f}{estimado:>10.0f}"
              f"{estimado - f['Minutos']:>+8.0f}{f['metros_por_hora']:>15.0f}")
    print()
    parrafo(f"La última columna es la velocidad de subida, en metros de desnivel por hora. En "
            f"estas ocho salidas va de {top['metros_por_hora'].min():.0f} a "
            f"{top['metros_por_hora'].max():.0f}. Todas cuentan igual como carrera con desnivel "
            "para el modelo, pero unas fueron competición y otras rodaje. El promedio de una "
            "actividad no distingue el esfuerzo, y con pocas carreras de montaña la regresión no "
            "puede separar cuánto cuesta el terreno de cuánto se apretó. La solución es el Grade "
            "Adjusted Pace calculado punto a punto sobre la ruta, que requiere las series de "
            "Strava y está pendiente.")
    return factor


def seccion_variantes(df, factor):
    """Compara las variantes del modelo contra las dos referencias."""
    seccion(3, "COMPARACIÓN DE VARIANTES")
    datos = construir_dataset(df, factor)
    comun = datos.dropna(subset=["mejor_ritmo_previo", "mejor_ritmo"])
    parrafo(f"Hay {len(datos)} bloques con carrera válida y {len(comun)} de ellos tienen un "
            "bloque anterior consecutivo. Todas las variantes se miden sobre esos mismos "
            "bloques, porque comparar errores calculados sobre muestras distintas no dice "
            "nada. Las variantes D y E usan el factor de desnivel del método 3 de la sección "
            "anterior, que es el que añade menos distancia por cada subida y por tanto el más "
            "conservador. Ten en cuenta que estas variantes incluyen la montaña ajustada, "
            "mientras que el modelo en producción solo usa carreras llanas, así que sus cifras "
            "pueden diferir un poco.")

    y = comun["mejor_ritmo"]
    mae_media = float(np.abs(y - y.mean()).mean())
    mae_previa = mean_absolute_error(y, comun["mejor_ritmo_previo"])

    variantes = [
        ("A", "Modelo original. Ritmo medio del bloque, km por semana y horas",
         ["ritmo_medio_eq", "km_semana", "horas_semana"]),
        ("B", "Solo km por semana y horas, sin el ritmo medio",
         ["km_semana", "horas_semana"]),
        ("C0", "Solo el mejor ritmo del bloque anterior",
         ["mejor_ritmo_previo"]),
        ("C", "C0 más km por semana y horas",
         ["mejor_ritmo_previo", "km_semana", "horas_semana"]),
        ("D", "C0 más km equivalentes por semana y horas",
         ["mejor_ritmo_previo", "km_eq_semana", "horas_semana"]),
        ("E", "D más desnivel semanal",
         ["mejor_ritmo_previo", "km_eq_semana", "horas_semana", "desnivel_semana"]),
    ]

    print(f"  {'':<5}{'Qué predice el error':<72}{'R²':>6}{'MAE':>8}{'En segundos':>13}")
    print("  " + "-" * 104)
    print(f"  {'Ref':<5}{'Repetir el ritmo medio de todo el historial':<72}{0.0:>6.2f}"
          f"{mae_media:>8.3f}{segundos(mae_media):>13}")
    print(f"  {'Ref':<5}{'Repetir el mejor ritmo del bloque anterior tal cual':<72}"
          f"{r2_score(y, comun['mejor_ritmo_previo']):>6.2f}{mae_previa:>8.3f}{segundos(mae_previa):>13}")
    resultados = {}
    for clave, descripcion, columnas in variantes:
        r2, mae = evaluar(comun, columnas)
        resultados[clave] = (r2, mae)
        print(f"  {clave:<5}{descripcion:<72}{r2:>6.2f}{mae:>8.3f}{segundos(mae):>13}")
    print()

    mejor = min(resultados, key=lambda k: resultados[k][1])
    parrafo(f"La variante con menor error es la {mejor}, con {resultados[mejor][1]:.3f} min/km. "
            f"Supera a la referencia de repetir el bloque anterior, que da {mae_previa:.3f}, "
            f"{'sí' if resultados[mejor][1] < mae_previa else 'no'}.")
    if "C0" in resultados and "C" in resultados:
        mejora = 100 * (resultados["C0"][1] - resultados["C"][1]) / resultados["C0"][1]
        if mejora > 5:
            parrafo(f"Añadir kilómetros y horas a C0 reduce el error un {mejora:.0f} %, así que "
                    "el entrenamiento empieza a aportar información.")
        else:
            parrafo("Añadir kilómetros y horas a C0 no mejora el error de forma apreciable. "
                    "Eso no significa que entrenar no sirva. Significa que en este historial "
                    "el volumen casi no varía de un bloque a otro, y sin variación no hay "
                    "contraste con el que medir su efecto. Es una limitación de la muestra, "
                    "no una verdad sobre el entrenamiento.")


def seccion_produccion(df):
    """Muestra el modelo que usa la aplicación y la progresión histórica."""
    seccion(4, "MODELO EN PRODUCCIÓN")
    parrafo("Es el que usa la aplicación y está en modelo.py. Una sola variable, el mejor "
            "ritmo del bloque anterior, sobre carreras de 5 km o más con ritmo entre 3 y 10 "
            "min/km. La regresión no repite la marca anterior tal cual sino que la corrige "
            "hacia el nivel habitual. Si un bloque fue excepcionalmente bueno, el siguiente "
            "suele ser algo peor, y si fue excepcionalmente malo, algo mejor. Ese tirón hacia "
            "el promedio es lo que aprende la regresión y lo que baja el error frente a copiar "
            "el dato anterior.")
    entrenamiento = modelo.entrenar_modelo(df)
    if entrenamiento is None:
        print("  No hay mesociclos suficientes para entrenar el modelo.\n")
        return

    print(f"  Bloques de entrenamiento : {entrenamiento['n_mesociclos']}")
    print(f"  R² con Leave-One-Out     : {entrenamiento['r2']:.2f}")
    print(f"  Error medio del modelo   : {entrenamiento['mae']:.3f} min/km, {segundos(entrenamiento['mae'])}")
    print(f"  Repetir bloque anterior  : {entrenamiento['mae_marca_previa']:.3f} min/km, "
          f"{segundos(entrenamiento['mae_marca_previa'])}")
    print(f"  Repetir ritmo medio      : {entrenamiento['mae_baseline']:.3f} min/km, "
          f"{segundos(entrenamiento['mae_baseline'])}")
    mejora = 100 * (1 - entrenamiento["mae"] / entrenamiento["mae_marca_previa"])
    print(f"  Mejora sobre repetir el bloque anterior: {mejora:.0f} %\n")

    metricas = modelo.metricas_ultimo_bloque(df)
    if metricas is not None:
        ritmo = modelo.predecir_ritmo(entrenamiento, metricas)
        print(f"  Mejor ritmo del bloque anterior : {metricas['mejor_ritmo_previo']:.2f} min/km")
        print(f"  Ritmo predicho para este bloque : {ritmo:.2f} min/km\n")

    parrafo("El verificador de viabilidad compara una meta contra lo que el historial dice que "
            "sueles mejorar. Esta tabla mide, para cada par de bloques separados por esa "
            "cantidad de meses, cuánto bajó o subió el mejor ritmo, y promedia.")
    print(f"  {'Plazo':<22}{'Mejora típica':>16}{'Pares medidos':>16}")
    for meses in (1, 2, 3, 6):
        progresion = modelo.progresion_historica(entrenamiento["datos"], meses)
        if progresion:
            signo = progresion["mejora_media_min_km"]
            texto = f"{abs(signo) * 60:.0f} s/km " + ("más rápido" if signo > 0 else "más lento")
            print(f"  {f'{meses} mesociclo' + ('s' if meses > 1 else ''):<22}{texto:>16}{progresion['pares']:>16}")
        else:
            print(f"  {f'{meses} mesociclos':<22}{'sin pares suficientes':>16}")
    print()


def cierre():
    """Indica qué cambiaría las conclusiones."""
    seccion(5, "QUÉ PODRÍA CAMBIAR ESTAS CONCLUSIONES")
    parrafo("Hay tres cosas que reabrirían la comparación. La primera es tener más variación de "
            "volumen entre bloques, por ejemplo meses de descarga o de carga fuerte, que es lo "
            "que le falta al historial para que kilómetros y horas se puedan medir. La segunda "
            "es incorporar las series de Strava punto a punto, que permiten calcular el Grade "
            "Adjusted Pace real y dejar de promediar la montaña. La tercera es acumular más "
            "bloques consecutivos, porque con unas pocas decenas de casos cualquier "
            "conclusión es frágil. Vuelve a ejecutar este informe cuando alguna de las tres "
            "cambie y compara con la ejecución anterior.")


def main():
    df = backend.cargar_y_procesar_datos()
    if df is None or df.empty:
        print("No se pudieron cargar los datos.")
        return

    introduccion()
    seccion_modelo_original(df)
    factor = seccion_desnivel(df)
    seccion_variantes(df, factor)
    seccion_produccion(df)
    cierre()


if __name__ == "__main__":
    main()
