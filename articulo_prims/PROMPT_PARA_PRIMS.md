# Prompt para preparar el artículo

Usa todos los materiales de esta carpeta para redactar un artículo científico
en español sobre CABRIALES. El texto debe ser autocontenido, técnicamente
preciso y separar claramente la metodología computacional de la interpretación
física.

Estructura solicitada:

1. Título.
2. Resumen y palabras clave.
3. Introducción y motivación de la muografía volcánica.
4. Arquitectura reproducible de CABRIALES.
5. Geometría DEM, aceptación angular y puntos P1/P2/P4/P5.
6. Generación de flujo CNF y normalización por metro cuadrado.
7. Longitud de roca, pérdida de energía y energía crítica.
8. Kernel empírico de multiple Coulomb scattering.
9. Migración angular interna.
10. Modelo espacial de in-scattering externo.
11. Validaciones, resultados de 90 días y discusión.
12. Limitaciones, trabajo futuro y conclusiones.

Reglas físicas que no deben violarse:

- No describir MCS como rebote o reflexión en la superficie del volcán.
- Distinguir dirección inicial externa de dirección final angularmente aceptada.
- No afirmar que el cálculo actual verifica la intersección con un detector.
- No presentar el escalado por superficie DEM como tasa instrumental.
- Explicar que el flujo CNF se genera para 1 m2 y que ampliar el área de
  inyección multiplica el número ideal de muones incidentes.
- Indicar que las áreas sumadas por punto no son necesariamente una superficie
  física única sin solapamiento.
- Mantener separados los 342 eventos MC aceptados de los conteos escalados por
  área.
- Reportar la incertidumbre Monte Carlo de cada punto.

Usa las figuras de `imagenes/` y propón pies de figura informativos. Prioriza:

- `01_geometria/dem_fans.png`;
- los cuatro muogramas de `02_muogramas_filtrados/`;
- las comparaciones de `03_event_mc/`;
- origen externo, resultado final y primer contacto con roca de
  `04_in_scattering/`.

Usa `resultados/four_point_summary.json` y los resúmenes por punto como fuente
numérica. Consulta `codigo/` para describir el algoritmo, los parámetros por
defecto y la reproducibilidad. No inventes parámetros ausentes ni atribuyas una
geometría instrumental que el modelo todavía no contiene.
