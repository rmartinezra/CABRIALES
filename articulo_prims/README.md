# Materiales CABRIALES para articulo PRIMS

Esta carpeta reúne una instantánea curada de la documentación, el código, las
figuras y los resultados resumidos necesarios para preparar un artículo sobre
CABRIALES. No contiene caches cinemáticos, archivos SHW, DEM, pesos duplicados
del modelo CNF ni corridas completas.

La fuente canónica sigue siendo el repositorio CABRIALES. Los archivos de esta
carpeta están pensados para lectura, carga en una herramienta editorial y
trazabilidad de las figuras.

## Contenido

```text
articulo_prims/
├── README.md
├── PROMPT_PARA_PRIMS.md
├── documentacion/
│   ├── README_CABRIALES.md
│   ├── USO_RAPIDO.md
│   ├── KERNEL_FULL_TAIL.md
│   └── README_MUON_CNF_TOOLKIT.md
├── codigo/
│   ├── cabriales.py
│   ├── orquestador_machin.py
│   ├── validar_corrida.py
│   ├── requirements.txt
│   ├── modulos/
│   ├── cnf/
│   ├── geant4/
│   └── utilidades_flujo/
├── imagenes/
│   ├── 00_readme/
│   ├── 01_geometria/
│   ├── 02_muogramas_filtrados/
│   ├── 03_event_mc/
│   └── 04_in_scattering/{P1,P2,P4,P5}/
└── resultados/
    ├── four_point_summary.json
    ├── four_point_summary.csv
    └── {P1,P2,P4,P5}/spatial_in_scattering_summary.json
```

## Figuras recomendadas

### Geometría y campo de visión

![DEM y abanicos](imagenes/01_geometria/dem_fans.png)

La carpeta `imagenes/01_geometria/` incluye además el FOV individual de cada
punto de observación.

### Muograma filtrado

![Muograma filtrado P1](imagenes/02_muogramas_filtrados/inside_volcano_filtered_P1_linear.png)

Los mapas equivalentes de P2, P4 y P5 están en la misma carpeta.

### Migración angular interna

![Event MC P1](imagenes/03_event_mc/event_mc_inside_source_smearing_binned_bin2p50deg_comparison_P1.png)

Estas figuras corresponden al event-by-event MC empírico con presentación
rebineada a 2.5 grados.

### In-scattering externo

| Dirección externa inicial | Dirección final aceptada |
|---|---|
| ![Origen externo P1](imagenes/04_in_scattering/P1/spatial_source_external_map.png) | ![Mapa final P1](imagenes/04_in_scattering/P1/spatial_final_accepted_map.png) |

Para cada punto también se incluyen el primer contacto con roca, la superficie
volcánica muestreada y el histograma de longitud de roca de los aceptados.

## Resultados resumidos de 90 días

**Trazabilidad:** las cifras y mapas de esta sección pertenecen a la corrida
anterior y conservan en sus JSON la ruta del kernel usado entonces. El modelo
híbrido full-tail actual pasó pruebas cortas, pero estos resultados de 90 días
todavía no se han regenerado con él. Véase
`documentacion/KERNEL_FULL_TAIL.md`.

| Punto | Aceptados MC | Área efectiva ideal | Escalado ideal por día | Error relativo MC |
|---|---:|---:|---:|---:|
| P1 | 124 | 1.5600 km2 | 2,149,333.33 | 8.98% |
| P2 | 93 | 2.5475 km2 | 2,632,416.67 | 10.37% |
| P4 | 65 | 3.7050 km2 | 2,675,833.33 | 12.40% |
| P5 | 97 | 1.6375 km2 | 1,764,861.11 | 10.15% |
| Total | 379 | 9.4500 km2 | 9,222,444.44 | - |

Las áreas de los cuatro puntos se suman como superficies objetivo específicas
de cada observador. No representan necesariamente un área física única y sin
solapamiento.

## Interpretación física obligatoria

El estudio distingue dos problemas:

1. Migración interna: dirección inicialmente aceptada que cambia a otro píxel
   aceptado por MCS.
2. In-scattering externo: dirección inicialmente fuera de la aceptación que
   atraviesa roca y termina con una dirección angular aceptada.

El segundo cálculo es un diagnóstico espacial ideal sobre superficie DEM. No
es reflexión ni rebote. El muón pierde energía y se deflecta por multiple
Coulomb scattering dentro de la roca.

El escalado por área **no es una tasa del detector**. Para convertirlo en una
predicción instrumental faltan, como mínimo, área física, orientación,
eficiencia e intersección espacial con el detector.

## Código

`codigo/` conserva el único orquestador, la interfaz simple, el validador y los
módulos físicos. El camino reproducible completo del repositorio original es:

```bash
python3 cabriales.py full --force
```

`codigo/cnf/` contiene los dos ejecutables del generador CNF y sus dependencias.
El peso `model.pt` no se duplica aquí; se encuentra en el repositorio principal
bajo `herramientas/muon-cnf-toolkit/model.pt`.

`codigo/geant4/` conserva el generador de casos cercanos al umbral usado para
estudios auxiliares del scattering. `codigo/utilidades_flujo/` reúne los scripts
de conversión y comparación angular de entradas SHW.

El archivo binario del kernel híbrido no se duplica en esta carpeta editorial.
Su nombre, procedencia, soporte y SHA-256 están documentados en
`documentacion/KERNEL_FULL_TAIL.md`; el modelo canónico vive en
`modulos/hybrid_empirical_kernel_library.npz` del repositorio principal.

## Orden sugerido para el artículo

1. Motivación de la muografía volcánica y contaminación angular.
2. Geometría DEM y aceptación de P1/P2/P4/P5.
3. Flujo CNF, normalización de 1 m2 y cache cinemático.
4. Longitud de roca, pérdida de energía y energía crítica.
5. Kernel empírico de MCS y migración angular interna.
6. Modelo espacial de in-scattering externo.
7. Validaciones numéricas y sensibilidad.
8. Resultados de 90 días para los cuatro puntos.
9. Limitaciones instrumentales y trabajo futuro.
