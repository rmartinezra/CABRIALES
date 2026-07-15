# Uso rápido de CABRIALES

## 1. Preparación

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

El framework principal usa `cabriales.py`. `orquestador_machin.py` es el único
motor interno y queda disponible para configuraciones avanzadas.

## 2. Prueba rápida

```bash
python3 cabriales.py smoke --force
```

Este comando ejecuta el perfil pequeño de Bariloche y valida sus salidas. Es la
primera prueba recomendada después de instalar o actualizar el repositorio.

Verificar independientemente el kernel empírico híbrido:

```bash
python3 cabriales.py kernel-smoke
```

El pipeline usa por defecto `modulos/hybrid_empirical_kernel_library.npz` con
despacho híbrido: `tail-aware` dentro del dominio full-tail cercano al umbral y
core empírico para el dominio amplio de energía. El soporte común es de
`-1600` a `1600 mrad` y el umbral es cero.

## 3. Corrida completa de 90 días

```bash
python3 cabriales.py full --force
```

Incluye:

- geometría y FOV para `P1`, `P2`, `P4` y `P5`;
- longitud de roca y energía crítica;
- fast-cache y mapas theta-phi filtrados;
- kernel empírico, smearing y event-by-event MC;
- campaña espacial de in-scattering con 8 workers por punto;
- validación del pipeline y del resumen de background.

La salida predeterminada es:

```text
run_machin90dia_allpoints_full/
```

Antes de reemplazar una corrida, revisar los comandos:

```bash
python3 cabriales.py full --dry-run
```

`--force` elimina la salida correspondiente antes de regenerarla. No se activa
de forma implícita.

## 4. Cache cinemático

El comando `full` busca automáticamente `machin90dia_kinematic_cache` en:

1. `CABRIALES_90D_CACHE`;
2. `data/cache/machin90dia_kinematic_cache`;
3. `../CNF/muon-cnf-toolkit/machin90dia_kinematic_cache`.

Para usar otra ubicación:

```bash
export CABRIALES_90D_CACHE=/ruta/al/machin90dia_kinematic_cache
python3 cabriales.py full --force
```

También puede pasarse `--kinematic-cache RUTA` en la línea de comandos.

## 5. Progreso

Las etapas muestran `START`, `RUNNING`, `PROGRESS`, `OK` o `ERROR`. El detalle
completo va a los logs de la corrida. El latido predeterminado aparece cada 30
segundos:

```bash
python3 cabriales.py full --status-interval-s 60 --force
```

Durante el background se informa el número de chunks terminados para cada
punto. Una falla indica el nombre de la etapa y el log que debe revisarse.

## 6. Ejecuciones parciales

Solo pipeline de 90 días:

```bash
python3 cabriales.py machin90d --force
```

Solo background espacial:

```bash
python3 cabriales.py background90d --force
```

Reutilizar puntos de background ya reducidos:

```bash
python3 cabriales.py background90d --continue-on-existing
```

Validar una corrida existente:

```bash
python3 cabriales.py validate run_machin90dia_allpoints_full
```

Ayuda avanzada:

```bash
python3 orquestador_machin.py --help
python3 modulos/16_run_spatial_in_scattering_4points.py --help
```

## 7. Generar flujo con CNF

El generador opcional está aislado en:

```text
herramientas/muon-cnf-toolkit/
```

No se ejecuta automáticamente desde CABRIALES. Instalar sus dependencias en su
propio entorno:

```bash
cd herramientas/muon-cnf-toolkit
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Para corridas largas, generar directamente el cache:

```bash
./main-cnf-cabriales-cache.py \
  --days 90 \
  --height ALTITUD_M \
  --bx BX_MICROTESLA \
  --bz BZ_MICROTESLA \
  --device auto \
  --write-workers 6 \
  --output-dir ../../data/cache/machin90dia_kinematic_cache
```

El toolkit también puede generar `.shw` con `main-cnf.py`. Ver su README local
para parámetros, formato y ejemplos.
