#!/usr/bin/env python3
"""
generar_datos.py — Generador del conjunto de datos del Laboratorio 3
Analítica de Datos — FISI, UNMSM — Prof. Juan Gamarra Moreno

Genera el histórico transaccional sintético de una cadena de retail peruana.
El conjunto incorpora, de forma deliberada, problemas de calidad de datos que
el estudiante debe descubrir durante el laboratorio.

Uso:
    python generar_datos.py                    # 500 000 filas (por defecto)
    python generar_datos.py --filas 5000000    # 5 millones de filas
    python generar_datos.py --filas 2000000 --salida ventas_grande.csv

La semilla es fija: dos ejecuciones con el mismo número de filas producen
exactamente el mismo archivo.
"""

import argparse
import csv
import gzip
import numpy as np

SEMILLA = 20262

REGIONES = ["LIMA", "AREQUIPA", "LA LIBERTAD", "PIURA", "CUSCO", "LAMBAYEQUE"]
PESO_REGION = [0.52, 0.13, 0.11, 0.10, 0.08, 0.06]

CATEGORIAS = [
    "ABARROTES", "BEBIDAS", "LACTEOS", "CARNES", "FRUTAS Y VERDURAS",
    "LIMPIEZA", "CUIDADO PERSONAL", "PANADERIA", "CONGELADOS", "MASCOTAS",
]
PESO_CATEGORIA = [0.20, 0.14, 0.11, 0.10, 0.13, 0.09, 0.08, 0.07, 0.05, 0.03]

# "MASCOTAS" se incorpora recién en 2024: evolución del catálogo
CATEGORIA_TARDIA = "MASCOTAS"
ANIO_ALTA_CATEGORIA = 2024

MEDIOS_PAGO = ["EFECTIVO", "TARJETA_DEBITO", "TARJETA_CREDITO", "BILLETERA_MOVIL"]

# Precio base por categoría (soles) y dispersión lognormal
PRECIO_BASE = {
    "ABARROTES": 8.50, "BEBIDAS": 6.20, "LACTEOS": 7.80, "CARNES": 24.00,
    "FRUTAS Y VERDURAS": 5.40, "LIMPIEZA": 12.30, "CUIDADO PERSONAL": 18.70,
    "PANADERIA": 4.10, "CONGELADOS": 15.60, "MASCOTAS": 32.00,
}

# Mes con incidente de pérdida de datos (volumen anómalamente bajo)
MES_INCIDENTE = (2023, 9)

INICIO = np.datetime64("2022-01-01T00:00:00")
FIN = np.datetime64("2026-07-31T23:59:59")

CABECERA = [
    "id_ticket", "fecha_hora", "id_tienda", "region", "id_producto",
    "categoria", "cantidad", "precio_unitario", "importe", "id_cliente",
    "medio_pago",
]


def generar(n_filas, ruta_salida, comprimir=True):
    rng = np.random.default_rng(SEMILLA)

    # ---------- Fechas ----------
    segundos_totales = int((FIN - INICIO) / np.timedelta64(1, "s"))
    offsets = rng.integers(0, segundos_totales, size=n_filas)
    fechas = INICIO + offsets.astype("timedelta64[s]")
    anios = fechas.astype("datetime64[Y]").astype(int) + 1970
    meses = fechas.astype("datetime64[M]").astype(int) % 12 + 1

    # Incidente: se elimina el 88 % de las filas del mes afectado
    mask_incidente = (anios == MES_INCIDENTE[0]) & (meses == MES_INCIDENTE[1])
    sobrevive = rng.random(n_filas) > 0.88
    conservar = ~mask_incidente | sobrevive

    # ---------- Dimensiones ----------
    region = rng.choice(REGIONES, size=n_filas, p=PESO_REGION)
    categoria = rng.choice(CATEGORIAS, size=n_filas, p=PESO_CATEGORIA)

    # La categoría tardía no existe antes de 2024: se reasigna
    mask_tardia = (categoria == CATEGORIA_TARDIA) & (anios < ANIO_ALTA_CATEGORIA)
    categoria[mask_tardia] = rng.choice(
        CATEGORIAS[:-1], size=int(mask_tardia.sum()),
        p=np.array(PESO_CATEGORIA[:-1]) / sum(PESO_CATEGORIA[:-1]))

    id_tienda = np.array([f"T{n:03d}" for n in rng.integers(1, 121, size=n_filas)])
    id_producto = np.array([f"P{n:05d}" for n in rng.integers(1, 8501, size=n_filas)])

    # ---------- Cantidad ----------
    cantidad = rng.integers(1, 9, size=n_filas)

    # ---------- Precio con deriva temporal (inflación + cambio de canasta) ----------
    base = np.array([PRECIO_BASE[c] for c in categoria])
    anios_transcurridos = (fechas - INICIO) / np.timedelta64(365, "D")
    factor_deriva = 1.0 + 0.18 * anios_transcurridos          # deriva de precios
    ruido = rng.lognormal(mean=0.0, sigma=0.42, size=n_filas)
    precio_unitario = np.round(base * factor_deriva * ruido, 2)
    precio_unitario = np.clip(precio_unitario, 0.50, None)

    importe = np.round(precio_unitario * cantidad, 2)

    # ---------- Cliente (nulo cuando el ticket es anónimo) ----------
    id_cliente = np.array([f"C{n:07d}" for n in rng.integers(1, 480001, size=n_filas)],
                          dtype=object)
    anonimo = rng.random(n_filas) < 0.083
    id_cliente[anonimo] = None

    # ---------- Medio de pago (con nulos) ----------
    medio_pago = rng.choice(MEDIOS_PAGO, size=n_filas,
                            p=[0.34, 0.27, 0.21, 0.18]).astype(object)
    medio_pago[rng.random(n_filas) < 0.012] = None

    id_ticket = np.array([f"TK{n:010d}" for n in range(1, n_filas + 1)], dtype=object)

    # ================= PROBLEMAS DE CALIDAD DELIBERADOS =================

    def indices(prob):
        return np.flatnonzero(rng.random(n_filas) < prob)

    # (1) Valor centinela en precio: 999999.0 marca "precio no disponible"
    idx = indices(0.0006)
    precio_unitario[idx] = 999999.0
    importe[idx] = np.round(999999.0 * cantidad[idx], 2)

    # (2) Devoluciones registradas con importe negativo
    idx = indices(0.0011)
    importe[idx] = -np.abs(importe[idx])

    # (3) Cantidades absurdas (error de digitación)
    idx = indices(0.0005)
    cantidad[idx] = rng.integers(600, 5000, size=idx.size)
    importe[idx] = np.round(precio_unitario[idx] * cantidad[idx], 2)

    # (4) Inconsistencia: importe != cantidad * precio_unitario
    idx = indices(0.0031)
    importe[idx] = np.round(importe[idx] * rng.uniform(1.10, 1.60, size=idx.size), 2)

    # (5) Región escrita de forma inconsistente
    idx = indices(0.0125)
    variantes = {"LIMA": ["Lima", " LIMA", "lima ", "LIMA "],
                 "AREQUIPA": ["Arequipa", "AREQUIPA "],
                 "PIURA": ["Piura", " Piura"],
                 "CUSCO": ["Cusco", "CUZCO"],
                 "LA LIBERTAD": ["La Libertad", "LA  LIBERTAD"],
                 "LAMBAYEQUE": ["Lambayeque", "LAMBAYEQUE "]}
    region = region.astype(object)
    for i in idx:
        opciones = variantes[region[i]]
        region[i] = opciones[rng.integers(0, len(opciones))]

    # (6) Fechas futuras (error del reloj de una caja)
    idx = indices(0.00004)
    fechas = fechas.astype("datetime64[s]")
    fechas[idx] = np.datetime64("2027-03-15T10:30:00")

    # (7) Tickets duplicados exactos (reproceso de la carga)
    n_dup = int(n_filas * 0.0018)
    idx_dup = rng.integers(0, n_filas, size=n_dup)

    # ---------- Escritura ----------
    fechas_txt = np.datetime_as_string(fechas, unit="s")
    orden = np.argsort(offsets, kind="stable")

    abrir = (lambda p: gzip.open(p, "wt", newline="", encoding="utf-8")) if comprimir \
        else (lambda p: open(p, "w", newline="", encoding="utf-8"))

    escritas = 0
    with abrir(ruta_salida) as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(CABECERA)

        def fila(i):
            return [
                id_ticket[i], fechas_txt[i].replace("T", " "), id_tienda[i],
                region[i], id_producto[i], categoria[i], int(cantidad[i]),
                f"{precio_unitario[i]:.2f}", f"{importe[i]:.2f}",
                id_cliente[i] if id_cliente[i] is not None else "",
                medio_pago[i] if medio_pago[i] is not None else "",
            ]

        for i in orden:
            if not conservar[i]:
                continue
            w.writerow(fila(i))
            escritas += 1
        for i in idx_dup:                      # duplicados al final del archivo
            if conservar[i]:
                w.writerow(fila(i))
                escritas += 1

    return escritas


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generador de datos — Laboratorio 3")
    ap.add_argument("--filas", type=int, default=500_000,
                    help="número de filas a generar (por defecto 500000)")
    ap.add_argument("--salida", default="ventas_2022_2026.csv.gz",
                    help="ruta del archivo de salida")
    ap.add_argument("--sin-comprimir", action="store_true",
                    help="escribir CSV plano en lugar de .csv.gz")
    a = ap.parse_args()

    ruta = a.salida[:-3] if (a.sin_comprimir and a.salida.endswith(".gz")) else a.salida
    n = generar(a.filas, ruta, comprimir=not a.sin_comprimir)
    print(f"Generado: {ruta}  ({n:,} filas escritas)")
