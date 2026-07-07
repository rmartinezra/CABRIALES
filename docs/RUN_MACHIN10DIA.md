# Corrida validada: `machin10dia.tar.gz`

Fecha de corrida: 2026-07-07.

Entrada:

```text
data/machin10dia.tar.gz
```

Comando base usado:

```bash
python3 orquestador_machin.py \
  --scripts-dir auto \
  --hgt-dir auto \
  --range-file auto \
  --shw data/machin10dia.tar.gz \
  --shw-format cnf9 \
  --storage-profile compact \
  --outdir run_machin_10dia \
  --points P1 P2 P4 P5 \
  --rho 2.65 \
  --plot-source both \
  --inside-volcano-source both \
  --scattering-model both \
  --smearing-source both \
  --run-event-mc \
  --event-mc-source both \
  --event-mc-source-mode all \
  --parallel-jobs 0 \
  --inside-filtered-workers 0
```

La corrida se completo y `validar_corrida.py` reporto:

```text
Salidas indexadas: 375
Salidas faltantes: 0
Alertas en logs: 0
```

Conteos principales:

```text
theta-phi raw:      P1=4270367, P2=4272620, P4=4272384, P5=4270391
theta-phi filtered: P1=3721816, P2=3352032, P4=848889,  P5=3500591
```

Event-MC conservo los conteos (`input_total == smeared_total`) para raw y
filtered.

## Diagnostico de rendimiento

La corrida funciona, pero el cuello de botella principal es:

```text
08c_event_mc_empirical_raw_all: ~776.6 min
```

El filtro tambien retuvo casi todo el archivo:

```text
P1: 156115203 muones
P2: 155727339 muones
P4: 152866129 muones
P5: 155892123 muones
```

Esto ocurre porque `--treat-out-of-grid-as-clear 1` conserva eventos fuera de
la grilla. Para corridas de produccion conviene evaluar si esa politica es la
correcta o si debe usarse `--treat-out-of-grid-as-clear 0` para reducir salida.

## Notas

- El `.tar.gz` CNF de 9 columnas fue leido correctamente.
- Las salidas filtradas `.shw.gz` fueron consumidas por plots, inside-volcano y
  event-MC.
- La GPU fue visible, pero no habia CuPy instalado; la corrida uso CPU.
