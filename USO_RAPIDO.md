# Uso rapido del pipeline Machin

## 0. Entrada recomendada

Para uso diario, la forma mas simple es `cabriales.py`. Es una capa delgada que
arma los comandos validados del orquestador y de los modulos de background, los
imprime y los ejecuta desde la raiz del repo.

Verificar que el framework esta conectado:

```bash
python3 cabriales.py smoke --force
```

Correr el pipeline Machin 90 dias con cache cinematico, fast-cache, smearing
empirico y event-by-event MC:

```bash
python3 cabriales.py machin90d --force
```

Correr el background espacial de in-scattering para P1/P2/P4/P5:

```bash
python3 cabriales.py background90d --force
```

Correr ambas cosas, pipeline y background:

```bash
python3 cabriales.py all90d --force
```

Puedes ver lo que se ejecutaria sin lanzar la corrida larga con `--dry-run`:

```bash
python3 cabriales.py all90d --dry-run
```

Las opciones avanzadas siguen disponibles en `orquestador_machin.py` y en los
modulos dentro de `modulos/`. `cabriales.py` no reemplaza esos scripts; solo hace
comodos los caminos validados.

## 1. Smoke test con Bariloche

Esta es la forma recomendada de verificar que todo el pipeline esta conectado.
No usa `--discard-upgoing`, porque el archivo `bariloche_5min.shw` tiene una
convencion de signo que deja el filtrado vacio con esa bandera.

```bash
python3 orquestador_machin.py \
  --profile bariloche-smoke \
  --outdir run_bariloche_smoke \
  --force
```

Ese perfil equivale a:

- `--scripts-dir auto` -> autodetecta `modulos/`
- `--hgt-dir auto` -> autodetecta `data/`
- `--shw data/bariloche_5min.shw`
- `--scattering-model both`
- `--smearing-source both`
- `--run-event-mc`
- `--event-mc-source both`
- workers automaticos

Validar la corrida:

```bash
python3 validar_corrida.py run_bariloche_smoke
```

## 2. Corrida Machin con archivo grande

```bash
python3 orquestador_machin.py \
  --scripts-dir auto \
  --hgt-dir auto \
  --range-file auto \
  --shw data/bga-2212-01_043200/bga-2212-01_043200.shw \
  --outdir run_machin_bga \
  --points P1 P2 P4 P5 \
  --rho 2.65 \
  --plot-source both \
  --inside-volcano-source both \
  --scattering-model both \
  --smearing-source both \
  --run-event-mc \
  --event-mc-source both
```

Usa `--discard-upgoing` solo si la convencion de `pz` del `.shw` lo requiere.
Con `bariloche_5min.shw` no debe usarse.

### Entrada CNF en tar comprimido

Si la entrada viene como `.tar`, `.tar.gz`, `.tgz`, `.tar.xz` o similar, el
pipeline toma automaticamente el primer archivo tipo `.shw` dentro del tar.
Para el formato CNF de 9 columnas:

```bash
python3 orquestador_machin.py \
  --scripts-dir auto \
  --hgt-dir auto \
  --range-file auto \
  --shw /ruta/machin10dia.tar.gz \
  --shw-format cnf9 \
  --storage-profile compact \
  --outdir run_machin_10dia
```

Si el tar trae varios `.shw`, especifica el miembro interno:

```bash
--shw-member inputs/machin10dia.shw
```

`--storage-profile compact` comprime los `.shw` filtrados como `.shw.gz`.
Tambien puedes controlarlo directamente con `--filtered-compression gz`.

### Ruta rapida con cache de eventos

Para corridas grandes donde solo necesitas la rama `filtered`, usa
`--fast-cache`. Esta ruta lee el `.shw` una sola vez, evita escribir
`04_filtered/*.shw.gz` y genera directamente:

- `04_event_cache/events_P*.npz`
- `05_plots/filtered/theta_phi_counts_P*.csv/png`
- `06_inside_volcano/filtered/P*/counts_inside_volcano_P*.csv`
- insumos para smearing y event-MC

Comando validado para P1 con el archivo de 12 horas de Bucaramanga:

```bash
python3 orquestador_machin.py \
  --scripts-dir modulos \
  --hgt-dir data \
  --range-file data/data_rock.dat \
  --shw data/bga-2212-01_043200/bga-2212-01_043200.shw \
  --outdir run_bucaramanga12h_p1_fastcache \
  --points P1 \
  --storage-profile compact \
  --fast-cache \
  --plot-source filtered \
  --inside-volcano-source filtered \
  --scattering-model empirical \
  --smearing-source filtered \
  --run-event-mc \
  --event-mc-source filtered \
  --event-mc-source-mode inside \
  --empirical-interp-method linear \
  --empirical-kernel-threshold 0.001 \
  --force
```

Resultado medido en esta maquina:

- tiempo total: `214.7 s`
- salida total: `32M`
- cache P1: `5.1M`
- mapa filtrado P1: `241335` cuentas
- event-MC inside: `1389 -> 1389` cuentas conservadas

La corrida equivalente anterior con `.shw.gz` filtrado ocupaba `467M`; solo
`04_filtered/*P1*.shw.gz` pesaba `438M`.

### Cache cinematico global

Para archivos muy grandes conviene separar el parseo del `.shw` de la fisica por
punto. El cache cinematico guarda, en chunks compactos, solo:

- `theta_deg`
- `phi_abs_deg`
- `total_E_GeV`
- bandera `pz_positive`
- tipo de muon

Construccion una sola vez para `machin10dia.tar.gz`:

```bash
python3 modulos/04_build_kinematic_cache.py \
  --shw data/machin10dia.tar.gz \
  --shw-format cnf9 \
  --out data/cache/machin10dia_kinematic_cache \
  --chunk-events 1000000 \
  --force
```

Luego la corrida rapida puede reutilizarlo:

```bash
python3 orquestador_machin.py \
  --scripts-dir modulos \
  --hgt-dir data \
  --range-file data/data_rock.dat \
  --shw data/machin10dia.tar.gz \
  --shw-format cnf9 \
  --kinematic-cache data/cache/machin10dia_kinematic_cache \
  --outdir run_machin10dia_p1_fastcache \
  --points P1 \
  --storage-profile compact \
  --fast-cache \
  --plot-source filtered \
  --inside-volcano-source filtered \
  --scattering-model empirical \
  --smearing-source filtered \
  --run-event-mc \
  --event-mc-source filtered \
  --event-mc-source-mode inside \
  --empirical-interp-method linear \
  --empirical-kernel-threshold 0.001 \
  --force
```

Si el cache no existe, el orquestador tambien puede construirlo automaticamente
cuando recibe `--kinematic-cache` y `--shw`.

## 3. GPU

WSL detecta la GPU con `nvidia-smi`, pero el pipeline actual usa NumPy/Pandas/SciPy.
Si no hay CuPy instalado, `--compute-device auto` informa el fallback y corre en CPU.

```bash
python3 orquestador_machin.py --profile bariloche-smoke --compute-device auto
```

El trabajo pesado actual esta optimizado por paralelismo entre puntos:

- `--parallel-jobs 0` autodetecta workers.
- `--inside-filtered-workers 0` autodetecta workers.
- `--event-mc-workers` usa `--parallel-jobs` si no se especifica.
