# Kernel empírico híbrido full-tail

## Modelo incorporado

- archivo de CABRIALES: `modulos/hybrid_empirical_kernel_library.npz`;
- SHA-256: `f39a0767b10688dea41a0f2bc0745382559fbb69ae20af1ddc17596742e07994`;
- simulaciones de entrada: 402 archivos ROOT;
- espesores de transporte: `1` a `1500 m`;
- núcleo: `-300` a `300 mrad`, bins de `1 mrad`;
- soporte full-tail: `-1600` a `1600 mrad`, bins de `1 mrad`;
- puntos de transporte full-tail: 181;
- método predeterminado: `tail-aware`.

Dentro del dominio full-tail cercano al umbral, el predictor usa transporte de
cuantiles para el cuerpo y mezcla histogramas medidos locales entre `250` y
`300 mrad`. Por encima de `300 mrad` usa la cola local medida para conservar
sucesos raros y hard scattering. Fuera de ese dominio CABRIALES usa la familia
core de rango energético amplio del mismo archivo, mediante el RBF lineal
validado al construir el modelo. Esta separación sigue la metadata original y
evita extrapolar colas cercanas al umbral a muones de alta energía. El umbral
de densidad aplicado es cero.

El transporte espacial usa el nodo nativo de `10 m`. Para consultas por encima
del rango energético medido del core, CABRIALES muestrea primero el histograma
empírico completo del vecino y luego aplica al ángulo la razón `1/(beta*p)`
entre la energía consultada y la energía de referencia. Esta corrección
mantiene los cuantiles, incluida la población de hard scattering, y evita
sobrestimar el ancho a altas energías. Los contadores del resumen separan pasos
full-tail, core, extrapolados y corregidos por momento.

El prefiltro CSDA también es de `10 m`: solo descarta muones incapaces de
completar el primer slab. No se conserva el antiguo corte de 100 m, porque
eliminaría trayectorias cortas de baja energía donde las colas pueden ser
relevantes.

## Prueba de integración

Comando:

```bash
python3 cabriales.py kernel-smoke
```

Resultado para `L=80 m` y `E=39.67 GeV`:

- normalización: `1.0`;
- `P(|theta| > 300 mrad) = 8.686064e-3`;
- `P(|theta| > 500 mrad) = 7.409548e-4`;
- `P(|theta| > 1000 mrad) = 1.308339e-6`;
- interpolación: bilineal sobre la grilla de transporte;
- política: cuerpo por cuantiles y cola por histograma local.

Una consulta de alta energía, por ejemplo `L=100 m` y `E=200 GeV`, usa el core
empírico amplio y produce un RMS cercano a `3.0 mrad`, sin reutilizar la cola
full-tail de energías cercanas al umbral.

## Estado de los resultados del artículo

Los JSON y figuras de `articulo_prims/` fueron regenerados el 15 de julio de
2026 con esta biblioteca híbrida. La corrida usó todo el cache de 90 días,
`sample_probability=1`, 10 workers y semilla base `12345`:

```bash
python3 cabriales.py full \
  --points P1 P2 P4 P5 \
  --workers 10 \
  --sample-probability 1.0 \
  --seed 12345 \
  --ray-step-m 10 \
  --kernel-energy-extrapolation momentum-scale \
  --force
```

El pipeline produjo 184 salidas indexadas, ninguna faltante y cero alertas en
logs. El background espacial leyó `1,363,053,739` eventos por punto y aceptó
245, 244, 296 y 231 eventos para P1, P2, P4 y P5, respectivamente: 1,016 en
total. El transporte acumuló 2,558,292 pasos full-tail, 47,935,842 pasos core y 6,032,066
pasos de alta energía corregidos por momento. Después de esa corrección se
registraron 6,096 kicks mayores de `300 mrad`, 262 mayores de `500 mrad` y ninguno
mayor de `1000 mrad`. Todos los puntos registraron cero pasos sin soporte del
kernel. Los resúmenes canónicos están en `resultados/` y conservan los
contadores de fallback, supervivencia, hard scattering, área efectiva e
incertidumbre Monte Carlo.

Con la normalización original de flujo sobre `1 m2`, las tasas superficiales
ideales son `2.722`, `2.711`, `3.289` y `2.567 muones/(m2 dia)` para P1, P2, P4
y P5. El promedio ponderado por las áreas objetivo es `2.914 muones/(m2 dia)`.
Estas tasas describen la superficie ideal de inyección y no incluyen geometría
ni eficiencia de detector. Al escalar también la exposición por el área efectiva,
los conteos MC de P1, P2, P4 y P5 corresponden a `4.98`, `3.05`, `2.10` y
`4.75 s` sobre sus superficies completas; el conteo combinado equivale a
`3.19 s` ponderados.
