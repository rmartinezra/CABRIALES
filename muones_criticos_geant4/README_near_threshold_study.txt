PRIMER ESTUDIO DE MUONES CERCA DEL UMBRAL
=========================================

Archivos:
- near_threshold_template.mac
- generate_near_threshold_macros.py

Uso recomendado:

python3 generate_near_threshold_macros.py \
  --ecrit-csv run_machin/03_ecrit/ecrit_table_P1.csv \
  --lengths 100 300 500 700 900 1100 1300 1500 \
  --factors 0.90 0.95 1.00 1.05 1.10 1.20 1.50 \
  --events 50000 \
  --outdir macros_near_threshold_P1

Luego:

cd macros_near_threshold_P1
bash run_all.sh

Qué mide cada corrida:
- Fracción de primarios transmitidos: aparece en el Run Summary de TestEm5.
- Energía cinética a la salida: histograma h10.
- Ángulo espacial: h12.
- Ángulo proyectado: h13.
- Posición y radio a la salida: h14 y h15.

Interpretación inicial:
- 0.90 y 0.95 Tcrit: región subcrítica; mide la cola de supervivencia.
- 1.00 a 1.20 Tcrit: población cercana al umbral.
- 1.50 Tcrit: control por encima del umbral.

Precauciones:
1. /gun/energy usa energía CINÉTICA. Por eso el generador usa Tcrit_GeV,
   no Ecrit_total_GeV.
2. El macro usa emstandard_opt4 y G4_SILICON_DIOXIDE para mantener
   consistencia con la biblioteca actual.
3. TestEm5, tal como viene por defecto, no constituye por sí solo una lista
   hadrónica completa. Antes de interpretar la curva de supervivencia como
   transporte total en roca, hay que verificar que tu aplicación incluya
   interacción muón-núcleo (G4MuonVDNuclearModel/G4EmExtraPhysics o equivalente).
4. Para el piloto usa 50 000 eventos. Para colas por debajo de 1e-4 será
   necesario aumentar a 1e6 o usar acumulación ponderada.
5. Repite una selección con cut = 0.1 cm para comprobar convergencia frente
   al valor piloto de 1 cm.
