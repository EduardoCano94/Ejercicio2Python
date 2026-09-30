import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Carga de datos
carpeta = Path(__file__).parent
datos = pd.read_excel(carpeta / "02-data-pcr.xlsx")

# Carpeta donde se guardan las figuras (sin la barra de ventana "Figure 1")
figuras = carpeta / "figuras"
figuras.mkdir(exist_ok=True)

print(datos.head())
print("\nNúmero de registros:", len(datos))

# Funciones auxiliares
def limites_iqr(serie):
    """Devuelve Q1, Q3, IQR y los límites de la regla del IQR."""
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    return q1, q3, iqr, q1 - 1.5 * iqr, q3 + 1.5 * iqr


def estadisticas(serie, nombre):
    """Imprime las medidas descriptivas de una variable numérica."""
    q1, q3, iqr, _, _ = limites_iqr(serie)

    print(f"\nEstadísticas de {nombre}")
    print("Media:", serie.mean())
    print("Mediana:", serie.median())
    print("Moda:", serie.mode()[0])
    print("Desviación estándar:", serie.std())
    print("Rango:", serie.max() - serie.min())
    print("Q1:", q1)
    print("Q3:", q3)
    print("IQR:", iqr)


def outliers(serie, nombre):
    """Imprime los valores atípicos según la regla del IQR."""
    q1, q3, iqr, lim_inf, lim_sup = limites_iqr(serie)
    atipicos = serie[(serie < lim_inf) | (serie > lim_sup)]

    print(f"\nOutliers de {nombre}")
    print("Q1:", q1)
    print("Q3:", q3)
    print("IQR:", iqr)
    print("Límite inferior:", lim_inf)
    print("Límite superior:", lim_sup)
    print("Número de outliers:", len(atipicos))
    print("Valores atípicos:", sorted(atipicos.unique().tolist()),
          "(conteo por valor:", atipicos.value_counts().to_dict(), ")")


def guardar_y_mostrar(fig, nombre_archivo):
    """Guarda la figura como PNG y la muestra."""
    fig.tight_layout()
    fig.savefig(figuras / nombre_archivo, dpi=150)
    plt.show()

# Parte 1: Análisis univariado
edad = datos["Age"]
ki67 = datos["Ki67"]

estadisticas(edad, "Edad")
estadisticas(ki67, "Ki67")

outliers(edad, "Edad")
outliers(ki67, "Ki67")

# Histograma de Edad
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(edad, bins=10, edgecolor="black")
ax.set_title("Distribución de la Edad")
ax.set_xlabel("Edad (años)")
ax.set_ylabel("Frecuencia")
guardar_y_mostrar(fig, "fig1_hist_edad.png")

# Histograma de Ki67
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(ki67, bins=10, edgecolor="black")
ax.set_title("Distribución del índice Ki67")
ax.set_xlabel("Ki67")
ax.set_ylabel("Frecuencia")
guardar_y_mostrar(fig, "fig2_hist_ki67.png")

# Tamaño del tumor (cT)
frecuencia_ct = datos["cT"].value_counts().sort_index()

print("\nFrecuencia del tamaño del tumor")
print(frecuencia_ct)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(frecuencia_ct.index, frecuencia_ct.values, edgecolor="black")
ax.set_title("Distribución del tamaño del tumor")
ax.set_xlabel("Tamaño del tumor")
ax.set_ylabel("Frecuencia")
guardar_y_mostrar(fig, "fig3_barras_ct.png")

# Parte 2: Comparación de grupos según pCR
conteo_pcr = datos["pCR"].value_counts()
porcentaje_pcr = datos["pCR"].value_counts(normalize=True) * 100

print("\nFrecuencia de pCR")
print(conteo_pcr)
print("\nPorcentaje de pCR")
print(porcentaje_pcr.round(1))

# Ki67 según pCR
print("\nKi67 según pCR (media y mediana)")
print(datos.groupby("pCR")["Ki67"].agg(["mean", "median"]))

fig, ax = plt.subplots(figsize=(8, 5))
datos.boxplot(column="Ki67", by="pCR", ax=ax)
ax.set_title("Ki67 según respuesta patológica completa")
fig.suptitle("")
ax.set_xlabel("Respuesta patológica completa (pCR)")
ax.set_ylabel("Ki67")
guardar_y_mostrar(fig, "fig4_boxplot_ki67_pcr.png")

# Edad según pCR
print("\nEdad según pCR (media y mediana)")
print(datos.groupby("pCR")["Age"].agg(["mean", "median"]))

fig, ax = plt.subplots(figsize=(8, 5))
datos.boxplot(column="Age", by="pCR", ax=ax)
ax.set_title("Edad según respuesta patológica completa")
fig.suptitle("")
ax.set_xlabel("Respuesta patológica completa (pCR)")
ax.set_ylabel("Edad (años)")
guardar_y_mostrar(fig, "fig5_boxplot_edad_pcr.png")

# Outliers dentro de cada grupo de pCR (lo que muestran los boxplots)
for variable in ["Ki67", "Age"]:
    for grupo, serie in datos.groupby("pCR")[variable]:
        outliers(serie, f"{variable} en el grupo pCR = {grupo}")

# Parte 3: Correlaciones

# cT se codifica como ordinal (T1=1 ... T4=4) y pCR como 0/1
datos["cT_num"] = datos["cT"].map({"T1": 1, "T2": 2, "T3": 3, "T4": 4})
datos["pCR_num"] = datos["pCR"].map({"No": 0, "Yes": 1})

variables = ["cT_num", "Age", "Ki67", "pCR_num"]

correlaciones = datos[variables].corr()  # Pearson
print("\nCorrelación de Pearson entre las variables")
print(correlaciones.round(3))

# Spearman como comparación (más adecuada para cT ordinal)
print("\nCorrelación de Spearman (comparación)")
print(datos[variables].corr(method="spearman").round(3))

# Las variables más asociadas con pCR
con_pcr = correlaciones["pCR_num"].drop("pCR_num")
ordenadas = con_pcr.reindex(con_pcr.abs().sort_values(ascending=False).index)

print("\nCorrelación con pCR ordenada por magnitud")
print(ordenadas.round(3))
print("Las 2 variables con mayor correlación con pCR:",
      ordenadas.index[:2].tolist())

# Relación entre Ki67 y pCR
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(datos["Ki67"], datos["pCR_num"])
ax.set_title("Relación entre Ki67 y pCR")
ax.set_xlabel("Ki67")
ax.set_ylabel("pCR (0 = No, 1 = Yes)")
guardar_y_mostrar(fig, "fig6_dispersion_ki67_pcr.png")

# Relación entre Edad y pCR
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(datos["Age"], datos["pCR_num"])
ax.set_title("Relación entre Edad y pCR")
ax.set_xlabel("Edad")
ax.set_ylabel("pCR (0 = No, 1 = Yes)")
guardar_y_mostrar(fig, "fig7_dispersion_edad_pcr.png")