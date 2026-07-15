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

Los JSON y figuras ya copiados en `articulo_prims/resultados/` corresponden a
la corrida de 90 días anterior y registran el kernel usado en esa corrida. La
integración descrita aquí solo tuvo una prueba corta: esos resultados no deben
presentarse como si hubieran sido recalculados con el modelo full-tail. Una
nueva corrida completa debe guardarse y compararse por separado.
