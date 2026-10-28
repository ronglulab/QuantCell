#!/bin/bash
# Reproduce the analyses and figures in the QuantCell paper. Usage: bash run_all.sh [stage ...]
# Stages (default: all, in this order): bm benchmarks_bm colon chl external segmentation spatial figures
#
# Set before running:
#   QC_DATA, QC_EXTERNAL, QC_OUT   data locations (see config.py)
#   PY        python with QuantCell's requirements + lightgbm 4.6.0 (README)        default: python
#   ASTIR_PY  python with astir 0.1.2                                               default: python
#   MAPS_PY   python with MAPS 1.0.0                                                default: python
#   ANNOSPAT  AnnoSpat executable (AnnoSpat 1.0.0)                                  default: AnnoSpat
#   RSCRIPT   Rscript with CELESTA, zeallot, Rmixmod, spdep (R 4.4.2)               default: Rscript
#   CELLPOSE_PY python with cellpose 4.0.6 and tifffile (segmentation stage only)    default: python
# Runtime: several days of CPU in total, dominated by hyperparameter tuning, the feature and marker analyses, MAPS, and the
# external model comparison. Each step is resumable where noted in its script.
set -euo pipefail
cd "$(dirname "$0")"; D=$(pwd)
PY=${PY:-python}; ASTIR_PY=${ASTIR_PY:-python}; MAPS_PY=${MAPS_PY:-python}; ANNOSPAT=${ANNOSPAT:-AnnoSpat}; RSCRIPT=${RSCRIPT:-Rscript}; CELLPOSE_PY=${CELLPOSE_PY:-python}
export QC_INPUTS="$D/inputs"
export QC_OUT=$($PY -c "import config; print(config.OUT)"); export QC_EXTERNAL=$($PY -c "import config; print(config.EXT)")
say(){ echo "[$(date '+%F %T')] $*"; }
annospat(){ # annospat DIR FIRST LAST N_TYPES
  local O=$1; local A="[$(printf '99,%.0s' $(seq 1 $(( $4 + 1 ))) | sed 's/,$//')]"
  ( cd "$O" && rm -rf outputdir && mkdir outputdir && s=$(date +%s) && $ANNOSPAT generateLabels -i "$O/marker_data.csv" -m "$O/marker_combos.csv" \
      -o "$O/outputdir" -f "$2" -l "$3" -r ROI -t "[99.9,99.999,70]" -a "$A" > "$O/run.log" 2>&1 && echo $(( $(date +%s) - s )) > "$O/time_elapsed_s.txt" ); }

stage_bm(){ say "bone marrow: input table, annotation, model selection, feature analyses"
  $PY bm_01_build_codex.py
  $PY bm_03_base_models.py --njobs 12
  $PY bm_04_tune_models.py --njobs 24                   # long
  $PY bm_05_tuned_cv.py --njobs 12
  QC_CLF=lgbm $PY bm_02_annotate.py 24
  QC_CLF=lgbm $PY bm_06_feature_analysis.py --mode included --njobs 12   # long
  QC_CLF=lgbm $PY bm_06_feature_analysis.py --mode excluded --njobs 12   # long
  QC_CLF=lgbm $PY bm_07_marker_knockout.py --njobs 16                    # long
  QC_CLF=lgbm $PY bm_07b_knockout_baseline.py
  QC_CLF=lgbm $PY bm_08_fdr_sweep_nested.py 16
  QC_CLF=lgbm $PY bm_08b_fdr_sweep_full.py 16
  $PY bm_02_annotate.py 24                              # random forest set (Fig. 4 bone marrow AnnoSpat/Astir/MAPS inputs use its split files)
}
stage_benchmarks_bm(){ say "bone marrow benchmarks: MAPS, Astir, AnnoSpat, CELESTA"
  local B=$QC_OUT/retrain/benchmarks
  $PY bm_10_benchmarks_prep.py
  $ASTIR_PY bm_10_astir.py
  annospat "$B/AnnoSpat" B220 Ter119 21
  OMP_NUM_THREADS=8 $MAPS_PY bm_10_maps.py
  $PY bm_10_celesta_prep.py; s=$(date +%s); $RSCRIPT bm_10_celesta.R; echo $(( $(date +%s) - s )) > "$B/CELESTA/time_elapsed_s.txt"
  $PY bm_11_benchmarks_score.py                         # also Supplementary Fig. 9
}
stage_colon(){ say "colon: QuantCell (random forest), MAPS, Astir, AnnoSpat, CELESTA"
  local O=$QC_OUT/celesta/colon/priors
  $PY colon_01_quantcell.py
  $PY colon_02_maps_prep.py; OMP_NUM_THREADS=8 $MAPS_PY colon_02_maps.py
  $PY colon_03_priors_prep.py; $ASTIR_PY colon_03_astir.py; annospat "$O/AnnoSpat" CD31 FOXP3 16
  $RSCRIPT colon_03_celesta.R
  $PY colon_04_runtimes.py
}
stage_chl(){ say "lymph node: cHL1, cHL CODEX, cHL2"
  for ds in chl1 codex; do
    local O=$QC_OUT/chl/${ds}_priors; mkdir -p "$O"; $PY chl_01_prep.py $ds | tee "$O/prep.log"
    read FIRST LAST NT < <(sed -n 's/.*AnnoSpat first\/last: \([^ ]*\) \([^ ]*\) | n types \([0-9]*\).*/\1 \2 \3/p' "$O/prep.log")
    $ASTIR_PY chl_02_astir.py $ds; annospat "$O/AnnoSpat" "$FIRST" "$LAST" "$NT"
    s=$(date +%s); CELESTA_B=$O/CELESTA CELESTA_PRIOR=prior_marker_info.csv CELESTA_OUT=per_sample $RSCRIPT chl_02_celesta.R; echo $(( $(date +%s) - s )) > "$O/CELESTA/time_elapsed_s.txt"
    OMP_NUM_THREADS=8 $MAPS_PY chl_02_maps.py $ds
  done
  local O=$QC_OUT/chl/chl2_priors; mkdir -p "$O"
  $PY chl2_01_maps_prep.py; OMP_NUM_THREADS=8 $MAPS_PY chl2_02_maps.py
  $PY chl2_01_priors_prep.py | tee "$O/prep.log"
  read FIRST LAST NT < <(sed -n 's/.*first\/last for AnnoSpat: \([^ ]*\) \([^ ]*\) | types \([0-9]*\).*/\1 \2 \3/p' "$O/prep.log")
  $ASTIR_PY chl2_02_astir.py; annospat "$O/AnnoSpat" "$FIRST" "$LAST" "$NT"
  s=$(date +%s); $RSCRIPT chl2_02_celesta.R; echo $(( $(date +%s) - s )) > "$O/CELESTA/time_elapsed_s.txt"
  for ds in bm colon chl1 chl2 codex; do $PY bench_01_fig4_data.py $ds; done
}
stage_external(){ say "ten-model comparison on the external datasets"; for ds in colon chl1 chl2 codex; do $PY ext_01_model_comparison.py $ds 12; done; }
stage_segmentation(){ say "membrane-based segmentation (Supplementary Fig. 3)"
  $CELLPOSE_PY seg_01_cellpose.py; $PY seg_02_cellpose_codex.py
  export QC_CLF=lgbm QC_NAME=QuantCellPaperCellpose
  $PY bm_02_annotate.py 24; $PY bm_06_feature_analysis.py --mode included --njobs 12; $PY bm_06_feature_analysis.py --mode excluded --njobs 12
  $PY bm_08_fdr_sweep_nested.py 16; $PY bm_08b_fdr_sweep_full.py 16; unset QC_NAME
}
stage_spatial(){ say "spatial features (Supplementary Fig. 11)"; for ds in bm colon chl1; do QC_CLF=lgbm $PY spatial_01_features.py $ds --njobs 12; done; }
stage_figures(){ say "figures and tables"
  $PY fig1_image_crop.py
  $PY fig2.py lgbm; QC_CLF=lgbm $PY fig3.py; $PY fig4_and_suppl12.py; $PY fig4_and_suppl12.py codex,chl2 fig4_suppl.png; $PY fig5.py; $PY table1.py
  [ -f "$($PY -c 'import config; print(config.STARDIST_GEOJSON)')" ] && $PY suppl3a.py 21.0 4.2 || say "Suppl Fig 3a skipped: no StarDist GeoJSON (QC_STARDIST_GEOJSON)"
  [ -f "$QC_OUT/fig1_image/suppl3a_wide_split.png" ] && $PY suppl3.py || say "Suppl Fig 3 skipped: panel a image missing"
  $PY suppl4.py; $PY suppl5.py; $PY suppl7.py; $PY suppl8.py; $PY suppl10.py; $PY suppl11.py
  $PY suppl_table1_data.py; $PY suppl_table1.py; $PY suppl_table2_data.py; $PY suppl_table2.py; $PY suppl_tables3_4_data.py; $PY suppl_tables3_4.py
}
STAGES=${*:-bm benchmarks_bm colon chl external segmentation spatial figures}
for st in $STAGES; do stage_$st; done; say "done: $STAGES"
