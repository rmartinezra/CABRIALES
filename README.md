# Pipeline directo de muografía para Volcán Machín

Este repositorio organiza una cadena reproducible para construir muogramas simulados del Volcán Machín a partir de:

1. un modelo topográfico DEM en archivos HGT,
2. una geometría de observación por punto (`P1`, `P2`, `P4`, `P5`),
3. una tabla de rango/energía de muones en roca,
4. un archivo `.shw` de ARTI/CORSIKA o CNF, plano o comprimido,
5. y una máscara angular de roca volcánica.

> Nota de estado: en esta carpeta los scripts de etapa viven en `modulos/`.
> El punto de entrada recomendado es `orquestador_machin.py`, que ahora delega
> al orquestador unificado con event-MC y autodetecta `modulos/`, `data/` y
> `modulos/empirical_kernel_library.npz`. Para una receta corta y actualizada,
> ver `USO_RAPIDO.md`.
>
> Para publicar el codigo en GitHub sin subir datos pesados, ver
> `docs/GITHUB_SETUP.md`. Para la corrida validada con `machin10dia.tar.gz`, ver
> `docs/RUN_MACHIN10DIA.md`.

Smoke test validado con el `.shw` pequeño:

```bash
python3 orquestador_machin.py \
  --profile bariloche-smoke \
  --outdir run_bariloche_smoke \
  --force
```

Validación:

```bash
python3 validar_corrida.py run_bariloche_smoke
```

Entradas soportadas para `--shw`: `.shw`, `.shw.gz`, `.shw.xz`, `.shw.bz2`,
`.tar`, `.tar.gz`, `.tgz`, `.tar.xz`, `.txz`, `.tar.bz2` y `.tbz2`. Para el
formato CNF de 9 columnas usa `--shw-format cnf9`; si el tar contiene varios
archivos, usa `--shw-member ruta/interna.shw`. Para ahorrar espacio en corridas
grandes, `--storage-profile compact` escribe los filtrados como `.shw.gz`.

La cadena completa ejecuta:

```text
01_puntos.py
    -> DEM, FOV, abanicos de visión, blocked_angles_P*.csv

02_longitud.py
    -> longitudes dentro de roca, rock_length_P*.csv

03_ecrit_heatmaps.py
    -> energía crítica por línea de visión, ecrit_table_P*.csv

06_filter_muons_by_ecrit.py
    -> filtrado de muones sobrevivientes por punto

05_plot_theta_phi.py
    -> mapas de conteos theta-phi

07_plot_counts_inside_volcano_geometry.py
    -> mapas de cuentas sólo dentro de la geometría angular del volcán
```

El script que coordina todo es:

```bash
orquestador_machin.py
```

python3 orquestador_machin.py \
  --scripts-dir . \
  --hgt-dir ./data \
  --range-file ./data/data_rock.dat \
  --shw ./data/bga_CNF_604800s.shw \
  --outdir ./run_machin_both_MC \
  --discard-upgoing \
  --scattering-model both \
  --smearing-source both \
  --smearing-stochastic \
  --empirical-kernel-library ./data/empirical_kernel_library.npz \
  --empirical-stochastic \
  --parallel-jobs 4 \
  --inside-filtered-workers 4


---

## 1. Estructura recomendada

Una estructura limpia de trabajo es:

```text
modelo_directo_muograma/
├── orquestador_machin.py
├── 01_puntos.py
├── 02_longitud.py
├── 03_ecrit_heatmaps.py
├── 05_plot_theta_phi.py
├── 06_filter_muons_by_ecrit.py
├── 07_plot_counts_inside_volcano_geometry.py
└── data/
    ├── N04W076.hgt
    ├── N04W075.hgt
    ├── muon_range_table.csv
    └── bga-2212-01_043200.shw
```

Los dos HGT requeridos son:

```text
N04W076.hgt
N04W075.hgt
```

El orquestador verifica que existan.

---

## 2. Ejecución completa recomendada

Para correr la cadena completa, incluyendo filtrado, mapas theta-phi y cuentas dentro de la geometría volcánica:

```bash
python3 orquestador_machin.py \
  --scripts-dir "$(pwd)" \
  --hgt-dir "$(pwd)/data" \
  --range-file "$(pwd)/data/muon_range_table.csv" \
  --shw "$(pwd)/data/bga-2212-01_043200.shw" \
  --outdir "$(pwd)/run_machin" \
  --points P1 P2 P4 P5 \
  --rho 2.65 \
  --plot-source both \
  --inside-volcano-source both \
  --force
```

Esto genera salidas para:

```text
raw       -> muones del .shw original
filtered  -> muones sobrevivientes después del corte por Ecrit(theta, phi)
```

---

## 3. Ejecución mínima sin archivo `.shw`

Si todavía no se quiere filtrar ni graficar muones, se puede correr sólo geometría, longitudes y energía crítica:

```bash
python3 orquestador_machin.py \
  --scripts-dir "$(pwd)" \
  --hgt-dir "$(pwd)/data" \
  --range-file "$(pwd)/data/muon_range_table.csv" \
  --outdir "$(pwd)/run_machin" \
  --points P1 P2 P4 P5 \
  --rho 2.65 \
  --force
```

En este modo no se ejecutan:

```text
04_filtered/
05_plots/
06_inside_volcano/
```

porque no hay `.shw`.

---

## 4. Ejecución sólo de una parte ya calculada

Si ya existen las salidas de geometría, longitudes, energía crítica y filtrado, y sólo se quiere regenerar la etapa de cuentas dentro del volcán:

```bash
python3 orquestador_machin.py \
  --scripts-dir "$(pwd)" \
  --hgt-dir "$(pwd)/data" \
  --range-file "$(pwd)/data/muon_range_table.csv" \
  --shw "$(pwd)/data/bga-2212-01_043200.shw" \
  --outdir "$(pwd)/run_machin" \
  --points P1 P2 P4 P5 \
  --rho 2.65 \
  --plot-source none \
  --inside-volcano-source filtered \
  --skip-geometry \
  --skip-lengths \
  --skip-ecrit \
  --skip-filter
```

Esto asume que ya existen, por ejemplo:

```text
run_machin/02_lengths/rock_length_P1.csv
run_machin/03_ecrit/ecrit_table_P1.csv
run_machin/04_filtered/bga-2212-01_043200_filtered_P1.shw
```

---

## 5. Estructura de salidas

El orquestador organiza las salidas así:

```text
run_machin/
├── 00_inputs/
│   ├── N04W076.hgt
│   ├── N04W075.hgt
│   ├── muon_range_table.csv
│   └── bga-2212-01_043200.shw
│
├── 01_geometry/
│   ├── dem_fans.png
│   ├── fov_P1.png
│   ├── blocked_angles_P1.csv
│   └── ...
│
├── 02_lengths/
│   ├── rock_length_P1.csv
│   ├── heatmap_P1.png
│   ├── summary.csv
│   └── ...
│
├── 03_ecrit/
│   ├── ecrit_table_P1.csv
│   ├── Tcrit_heatmap_P1.png
│   ├── Etotal_heatmap_P1.png
│   └── ...
│
├── 04_filtered/
│   ├── bga-2212-01_043200_filtered_P1.shw
│   └── ...
│
├── 05_plots/
│   ├── raw/
│   │   ├── theta_phi_counts_P1.png
│   │   └── theta_phi_counts_P1.csv
│   └── filtered/
│       ├── theta_phi_counts_P1.png
│       └── theta_phi_counts_P1.csv
│
├── 06_inside_volcano/
│   ├── raw/
│   │   └── P1/
│   │       ├── counts_inside_volcano_P1.png
│   │       ├── counts_inside_volcano_P1.csv
│   │       ├── dNdOmega_inside_volcano_P1.png
│   │       ├── dNdOmega_inside_volcano_P1.csv
│   │       └── inside_volcano_summary_P1.csv
│   └── filtered/
│       └── P1/
│           ├── counts_inside_volcano_P1.png
│           ├── counts_inside_volcano_P1.csv
│           ├── dNdOmega_inside_volcano_P1.png
│           ├── dNdOmega_inside_volcano_P1.csv
│           └── inside_volcano_summary_P1.csv
│
├── logs/
│   ├── 01_geometry.log
│   ├── 02_lengths.log
│   ├── 03_ecrit.log
│   ├── 04_filter_P1.log
│   ├── 05_plot_filtered_P1.log
│   └── ...
│
├── pipeline_outputs.csv
└── run_manifest.json
```

Archivos importantes:

```text
pipeline_outputs.csv
```

Índice de salidas generadas por etapa.

```text
run_manifest.json
```

Registro de parámetros, rutas resueltas, carpetas y estados de ejecución.

```text
logs/
```

Logs completos de cada etapa. Si algo falla, revisar primero el log indicado por el error.

---

## 6. Convenciones angulares

La cadena usa una convención cenital:

```text
theta = acos(pz / |p|)
```

donde:

```text
theta = 0°    -> dirección vertical hacia arriba
theta = 90°   -> dirección horizontal
theta > 90°   -> dirección por debajo del plano horizontal/local
```

La coordenada azimutal se calcula como:

```text
phi = atan2(py, px)
```

y luego se transforma a azimut relativo respecto a la dirección del punto hacia la cima del volcán.

En la etapa `07_plot_counts_inside_volcano_geometry.py`, por defecto sólo se grafican celdas con:

```text
0° <= theta <= 90°
```

porque no interesa mirar por debajo del piso.

---

## 7. Corrección por jacobiano angular

Los conteos crudos por píxel angular son:

```text
N(theta_i, phi_j)
```

Pero si se quiere una cantidad proporcional a intensidad diferencial por ángulo sólido, hay que dividir por:

```text
DeltaOmega_ij = DeltaPhi_j [cos(theta_low_i) - cos(theta_high_i)]
```

La etapa `07_plot_counts_inside_volcano_geometry.py` guarda ambas salidas:

```text
counts_inside_volcano_P1.png
```

Conteos crudos dentro de la geometría angular del volcán.

```text
dNdOmega_inside_volcano_P1.png
```

Conteos corregidos por ángulo sólido:

```text
N / DeltaOmega
```

Esta corrección no cambia la geometría, la longitud de roca ni el filtrado por energía crítica. Sólo cambia la normalización angular de los histogramas/mapas.

---

## 8. Significado físico de las etapas

### 8.1. Geometría y FOV

`01_puntos.py` genera la geometría angular de observación. Produce mapas del DEM, abanicos de visión y archivos como:

```text
blocked_angles_P1.csv
```

Estos archivos indican qué líneas de visión interceptan la topografía.

### 8.2. Longitud de roca

`02_longitud.py` calcula, para cada línea de visión:

```text
L(theta, phi)
```

donde `L` es la distancia recorrida dentro de roca.

Salida principal:

```text
rock_length_P1.csv
```

Esta es la salida más directa para definir la geometría angular del volcán. Una celda con:

```text
rock_length > 0
```

se considera una celda que atraviesa roca.

### 8.3. Energía crítica

`03_ecrit_heatmaps.py` transforma la longitud de roca en una energía crítica:

```text
Ecrit(theta, phi)
```

A mayor espesor de roca, mayor energía mínima necesaria para que un muón sobreviva.

Salida principal:

```text
ecrit_table_P1.csv
```

### 8.4. Filtrado de muones

`06_filter_muons_by_ecrit.py` lee el `.shw` y conserva los muones que cumplen:

```text
E_mu >= Ecrit(theta, phi)
```

por cada punto de observación.

Salida principal:

```text
bga-2212-01_043200_filtered_P1.shw
```

### 8.5. Mapas theta-phi

`05_plot_theta_phi.py` grafica conteos angularmente:

```text
theta_phi_counts_P1.png
theta_phi_counts_P1.csv
```

Puede graficar el archivo `.shw` crudo, el filtrado o ambos.

### 8.6. Cuentas dentro del volcán

`07_plot_counts_inside_volcano_geometry.py` toma el `.shw` crudo o filtrado y cuenta sólo los eventos que caen en celdas donde:

```text
rock_length > inside_mask_min
```

Por defecto:

```text
inside_mask_min = 0
```

---

## 9. Banderas del orquestador

### 9.1. Rutas principales

| Bandera | Tipo | Requerida | Default | Descripción |
|---|---:|---:|---:|---|
| `--scripts-dir` | ruta | no | `.` | Carpeta donde están los scripts `01`, `02`, `03`, `05`, `06` y opcionalmente `07`. |
| `--hgt-dir` | ruta | sí | — | Carpeta con los HGT requeridos: `N04W076.hgt` y `N04W075.hgt`. |
| `--range-file` | ruta | no | `None` | Tabla de rango/energía (`data_rock.dat` o `muon_range_table.csv`). Si no se da, se busca automáticamente. |
| `--shw` | ruta | no | `None` | Archivo `.shw` plano/comprimido o `.tar` con un `.shw` dentro. Si se omite, no se filtran ni grafican muones. |
| `--shw-format` | opción | no | `auto` | Formato de entrada: `auto`, `arti12` o `cnf9`. Usa `cnf9` para `CNFId energy theta px py pz h bx bz`. |
| `--shw-member` | texto | no | `None` | Miembro interno si `--shw` es un tar con varios archivos. |
| `--storage-profile` | opción | no | `normal` | `compact` activa compresión gzip de los `.shw` filtrados si no se indicó otra compresión. |
| `--filtered-compression` | opción | no | `none` | Compresión explícita para `04_filtered/*.shw`: `none`, `gz`, `xz` o `bz2`. |
| `--outdir` | ruta | no | `run_machin` | Carpeta raíz de salida. |

Ejemplo:

```bash
--scripts-dir "$(pwd)" \
--hgt-dir "$(pwd)/data" \
--range-file "$(pwd)/data/muon_range_table.csv" \
--shw "$(pwd)/data/bga-2212-01_043200.shw" \
--outdir "$(pwd)/run_machin"
```

---

### 9.2. Puntos y propiedades físicas

| Bandera | Tipo | Default | Descripción |
|---|---:|---:|---|
| `--points` | lista | `P1 P2 P4 P5` | Puntos de observación que se procesarán. Valores permitidos: `P1`, `P2`, `P4`, `P5`. |
| `--rho` | float | `2.65` | Densidad efectiva de roca en g/cm³ para calcular energía crítica. |

Ejemplo:

```bash
--points P1 P2 P4 P5 \
--rho 2.65
```

---

### 9.3. Filtro de muones por energía crítica

| Bandera | Tipo | Default | Descripción |
|---|---:|---:|---|
| `--tol-phi` | float | `0.51` | Tolerancia angular en grados para asociar el `phi` del muón con la grilla de `Ecrit`. |
| `--tol-theta` | float | `0.51` | Tolerancia angular en grados para asociar el `theta` del muón con la grilla de `Ecrit`. |
| `--treat-out-of-grid-as-clear` | entero `0/1` | `1` | Si vale `1`, conserva muones fuera de la grilla de `Ecrit`; si vale `0`, los descarta. |
| `--discard-upgoing` | flag | `False` | Pasa `--discard_upgoing` al filtro. En tus `.shw`, esta opción puede descartar todos los muones útiles porque muchos tienen `pz > 0`. Usarla con cuidado. |

Recomendación para los `.shw` que hemos probado:

```text
No usar --discard-upgoing
```

porque el archivo tiene los muones útiles con `pz > 0`.

---

### 9.4. Mapas theta-phi generales

| Bandera | Tipo | Default | Descripción |
|---|---:|---:|---|
| `--plot-source` | opción | `filtered` | Define qué `.shw` graficar con `05_plot_theta_phi.py`. Opciones: `none`, `raw`, `filtered`, `both`. |
| `--plot-theta-min` | float | `60.0` | Theta mínimo para los mapas generales. |
| `--plot-theta-max` | float | `90.0` | Theta máximo para los mapas generales. |
| `--plot-phi-min` | float | `-50.0` | Phi relativo mínimo para los mapas generales. |
| `--plot-phi-max` | float | `50.0` | Phi relativo máximo para los mapas generales. |
| `--bins-theta` | int | `60` | Número de bins en theta. |
| `--bins-phi` | int | `40` | Número de bins en phi relativo. |

Ejemplo para graficar sólo sobrevivientes:

```bash
--plot-source filtered
```

Ejemplo para graficar crudos y sobrevivientes:

```bash
--plot-source both
```

Ejemplo para no hacer mapas theta-phi:

```bash
--plot-source none
```

---

### 9.5. Cuentas dentro de la geometría del volcán

| Bandera | Tipo | Default | Descripción |
|---|---:|---:|---|
| `--inside-volcano-source` | opción | `none` | Ejecuta `07_plot_counts_inside_volcano_geometry.py`. Opciones: `none`, `raw`, `filtered`, `both`. |
| `--inside-mask-min` | float | `0.0` | Umbral para definir la máscara: una celda está dentro si `mask_col > inside_mask_min`. |
| `--inside-mask-col` | string | `None` | Columna usada para definir la máscara. Si se omite, el script intenta autodetectar una columna de longitud de roca. |

Uso recomendado:

```bash
--inside-volcano-source filtered \
--inside-mask-min 0
```

Para crudo y filtrado:

```bash
--inside-volcano-source both
```

La máscara usa, por defecto:

```text
rock_length_P*.csv
```

y define:

```text
dentro del volcán = rock_length > 0
```

El script `07` corta por defecto:

```text
0° <= theta <= 90°
```

para no graficar debajo del piso.

---

### 9.6. Control de ejecución

| Bandera | Tipo | Default | Descripción |
|---|---:|---:|---|
| `--force` | flag | `False` | Borra la carpeta `--outdir` antes de correr. Útil para una corrida limpia. |
| `--dry-run` | flag | `False` | No ejecuta los scripts; sólo escribe los comandos y logs. Sirve para revisar qué haría el orquestador. |
| `--skip-geometry` | flag | `False` | Omite `01_puntos.py`. Requiere que ya existan las salidas necesarias en `01_geometry/`. |
| `--skip-lengths` | flag | `False` | Omite `02_longitud.py`. Requiere que ya existan `rock_length_P*.csv`. |
| `--skip-ecrit` | flag | `False` | Omite `03_ecrit_heatmaps.py`. Requiere que ya existan `ecrit_table_P*.csv`. |
| `--skip-filter` | flag | `False` | Omite el filtrado por `Ecrit`. Requiere filtrados existentes si se piden mapas `filtered`. |
| `--skip-plots` | flag | `False` | Omite los mapas generales de `05_plot_theta_phi.py`. No omite la etapa `07` si `--inside-volcano-source` no es `none`. |

---

## 10. Comandos útiles

### 10.1. Ver ayuda del orquestador

```bash
python3 orquestador_machin.py --help
```

### 10.2. Revisar logs si algo falla

Ejemplo:

```bash
tail -n 80 run_machin/logs/04_filter_P1.log
```

Otro ejemplo:

```bash
tail -n 80 run_machin/logs/06_inside_volcano_filtered_P1.log
```

### 10.3. Revisar índice de salidas

```bash
column -s, -t run_machin/pipeline_outputs.csv | less -S
```

### 10.4. Revisar resumen de una etapa `07`

```bash
cat run_machin/06_inside_volcano/filtered/P1/inside_volcano_summary_P1.csv
```

---

## 11. Problemas frecuentes

### 11.1. Falta `02_longitud_fast.py`

Si estás usando una versión del orquestador que acepta `--lengths-script fast`, debes tener:

```text
02_longitud_fast.py
```

En la versión actual descrita aquí, el orquestador usa directamente:

```text
02_longitud.py
```

### 11.2. Error con `tqdm`

Si aparece algo como:

```text
AttributeError: 'range' object has no attribute 'update'
```

instala `tqdm`:

```bash
python3 -m pip install --user tqdm
```

o usa una versión parcheada de los scripts con fallback compatible.

### 11.3. Todos los muones desaparecen con `--discard-upgoing`

No uses esa bandera para los `.shw` donde los muones útiles tienen:

```text
pz > 0
```

En nuestras pruebas, activar esa bandera dejó cero candidatos.

### 11.4. Error `np.trapezoid`

Si usas el script de análisis angular con modelos y aparece:

```text
AttributeError: module 'numpy' has no attribute 'trapezoid'
```

cambia:

```python
np.trapezoid(...)
```

por:

```python
np.trapz(...)
```

Esto es un problema de versión de NumPy.

---

## 12. Qué archivos mirar primero

Para un resultado rápido de la cadena:

```text
run_machin/05_plots/filtered/theta_phi_counts_P1.png
```

Para ver sólo cuentas dentro del volcán:

```text
run_machin/06_inside_volcano/filtered/P1/counts_inside_volcano_P1.png
```

Para ver la versión corregida por ángulo sólido:

```text
run_machin/06_inside_volcano/filtered/P1/dNdOmega_inside_volcano_P1.png
```

Para revisar longitudes de roca:

```text
run_machin/02_lengths/rock_length_P1.csv
```

Para revisar energía crítica:

```text
run_machin/03_ecrit/ecrit_table_P1.csv
```

Para revisar los muones sobrevivientes:

```text
run_machin/04_filtered/bga-2212-01_043200_filtered_P1.shw
```

---

## 13. Recomendación de corrida base

Para una corrida limpia y completa, usaría:

```bash
python3 orquestador_machin.py \
  --scripts-dir "$(pwd)" \
  --hgt-dir "$(pwd)/data" \
  --range-file "$(pwd)/data/muon_range_table.csv" \
  --shw "$(pwd)/data/bga-2212-01_043200.shw" \
  --outdir "$(pwd)/run_machin" \
  --points P1 P2 P4 P5 \
  --rho 2.65 \
  --tol-phi 0.51 \
  --tol-theta 0.51 \
  --treat-out-of-grid-as-clear 1 \
  --plot-source both \
  --inside-volcano-source both \
  --inside-mask-min 0 \
  --plot-theta-min 60 \
  --plot-theta-max 90 \
  --plot-phi-min -50 \
  --plot-phi-max 50 \
  --bins-theta 60 \
  --bins-phi 40 \
  --force
```

No incluiría:

```bash
--discard-upgoing
```

salvo que se haya verificado explícitamente que esa convención corresponde al `.shw` usado.

---

## 14. Notas físicas importantes

1. El jacobiano angular se usa sólo para normalizar mapas/histogramas.  
   No debe entrar en el cálculo de longitud de roca.

2. La longitud de roca se calcula con la dirección angular del rayo.  
   Por eso la convención de `theta` sí importa para `L(theta, phi)`.

3. En la cadena actual, `theta` es cenital de forma consistente entre:
   - geometría,
   - longitud de roca,
   - energía crítica,
   - filtro de muones,
   - y mapas.

4. Para muografía de montaña, las direcciones más relevantes suelen estar cerca de:
   ```text
   theta ~ 90°
   ```
   porque son trayectorias casi horizontales.

5. La etapa `07` corta por defecto en:
   ```text
   theta <= 90°
   ```
   porque no interesa mirar por debajo del plano local.
