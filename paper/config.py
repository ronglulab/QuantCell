"""Paths used by every script in paper/. Set them with environment variables (recommended) or edit the defaults below.

QC_DATA      folder with the extracted Zenodo deposit (raw_image/, qupath_exports/, quantcell/)
QC_EXTERNAL  folder with the external datasets:
               colon/CRC_clusters_neighborhoods_markers.csv              (Schurch et al., Cell 2020)
               cHL_1_MIBI/annotation_csv/cHL1_MIBI.csv                     (MAPS, Shaban et al. 2024; Zenodo 10067010)
               cHL_2_MIBI/annotation_csv/1.csv ... 6.csv
               cHL_CODEX/annotation_csv/cHL_CODEX_annotation.csv
QC_OUT       folder for all intermediate results and figures (created if missing)
QC_STARDIST_GEOJSON  optional: QuPath GeoJSON export of the StarDist cells, only for Supplementary Fig. 3a
"""
import os
PAPER = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(PAPER)
CODE = os.path.join(REPO, 'Quantcell')                 # the QuantCell library
INPUTS = os.path.join(PAPER, 'inputs')                 # small published inputs shipped with the code (marker lists, priors)
DATA = os.path.abspath(os.environ.get('QC_DATA', os.path.join(REPO, 'data')))
EXT = os.path.abspath(os.environ.get('QC_EXTERNAL', os.path.join(REPO, 'external_data')))
OUT = os.path.abspath(os.environ.get('QC_OUT', os.path.join(REPO, 'output')))
QPTIFF = os.path.join(DATA, 'raw_image', 'QuantCell_bone_marrow_PhenoCycler.qptiff')
QUPATH = os.path.join(DATA, 'qupath_exports', 'full_panel')
COMBOS = os.path.join(DATA, 'quantcell', 'marker_combinations.json')
STARDIST_GEOJSON = os.environ.get('QC_STARDIST_GEOJSON', os.path.join(DATA, 'stardist_cells.geojson'))
os.makedirs(OUT, exist_ok=True)
