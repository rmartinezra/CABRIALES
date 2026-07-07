# Uso rapido del pipeline Machin

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
