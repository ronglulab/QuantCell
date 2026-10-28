# QuantCell

# Introduction
QuantCell is a machine learning based framework designed to enhance cell annotation in spatial omics data by integrating both qualitative and quantitative imaging profiles. It addresses significant challenges in analyzing complex tissues, such as those with limited molecular markers, overlapping marker expression, or rare cell types.

# Installation Guide
## Prerequisites for QuantCell
* Python 3.11.0
* NumPy 1.26.0
* pandas 1.5.3
* scikit-learn 1.5.0
* Matplotlib 3.7.1
* glob2 0.7
* Joblib 1.4.2
* tqdm 4.65.0
* LightGBM 4.6.0 (used by the tutorials; any scikit-learn compatible classifier can be used instead)

## Prerequisites for Replicating Paper Figures
* All of the prerequisites mentioned above
* SciPy 1.16.3
* seaborn 0.13.2
* ipykernel 7.1.0
* pyyaml 6.0
* astir 0.1.2
* MAPS 1.0.0
* AnnoSpat 1.0.0
* CELESTA (R package; R 4.4.2)

*ipykernel is needed to run the provided Jupyter notebooks but is not necessary for QuantCell itself

*QuantCell was tested using a Python environment with the prerequisite packages above on a dedicated server with Ubuntu 20.04.6 LTS. There are no other hardware or software dependencies but the authors make no guarantee it will work on every system.

To install the required packages into a new conda environment, you can use conda and pip:
(expected time: 1-3 minutes)
```
conda create -n "quantcell" python=3.11
conda activate quantcell
pip install numpy==1.26.0 pandas==1.5.3 scikit-learn==1.5.0 matplotlib==3.7.1 glob2==0.7 joblib==1.4.2 tqdm==4.65.0 lightgbm==4.6.0
pip install scipy==1.16.3 seaborn==0.13.2 pyyaml==6.0 ipykernel==7.1.0
```

## Running MAPS, Astir, AnnoSpat, and CELESTA to create comparison data
QuantCell is benchmarked against these four methods in the paper on the same datasets (scripts in `paper/`). Due to version incompatibilities, they need to be run in separate environments (CELESTA runs in R). Please see the associated GitHubs for instructions on how to install them:
* [MAPS](https://github.com/mahmoodlab/MAPS/)
* [astir](https://github.com/camlab-bioml/astir/)
* [AnnoSpat](https://github.com/faryabiLab/AnnoSpat/)
* [CELESTA](https://github.com/plevritis-lab/CELESTA)


# Input
QuantCell requires two primary types of input data:
1. Gene x cell expression matrix

(Optional) If QuPath is used to segment and annotate marker expression, QuantCell can automatically build the gene x cell matrix from the .csv marker files outputted (see `tutorial/00_QuPath_guide.md` and `02a_initialization_qupath.ipynb`). This is how the sample data is given. For this project and the dataset used, we used StarDist to segment the cells using a script in this repo (StarDist cell segmentation script_0.37_um_per_pixel.groovy). 
 - Once segmented, qualitative marker annotations can be included if they are annotated using QuPath following this tutorial: https://qupath.readthedocs.io/en/0.4/docs/tutorials/multiplex_analysis.html
 - Quantitative data (measurements and statistics) and qualitative data can be exported by following this QuPath tutorial: https://qupath.readthedocs.io/en/0.4/docs/tutorials/exporting_measurements.html  

__Important__: If you are providing your own Gene x Cell matrix (see `02b_initialization_custom_table.ipynb`), your column names must follow the format: MarkerName: Compartment: Measurement.
* __Compartments allowed:__ Nucleus, Cytoplasm, Membrane, Cell
* __Measurements allowed:__ Mean, Median, Min, Max, Std.Dev.
* Example: CD3: Nucleus: Mean, FoxP3: Cell: Median

Alternatively, provide a custom regex that matches columns with quantitative data you would like to use right after initialization of the quantcell_project and before `process_data()` (in `03_model_selection.ipynb` and `04_annotation.ipynb`).
* Add ```quantcell_project.feature_regex = "..." ```

&nbsp;

2. Marker combos for cell types
   These are currently implemented as a .json file which maps to a dictionary with the following format:
```
marker_combos = {
   'CellType1' : ['Marker1+', 'Marker2-', 'Marker3+'],
   'CellType2' : ['Marker2+', 'Marker3-'],
   'CellType3' : ['Marker4+', 'Marker3-'],
   ...
}
```
For best results, cell types should be defined by the minimum set of markers needed to differentiate them from other cells. Providing too many markers that are redundant can hurt performance and lead to too few annotated cells. Cells can only be assigned to one cell type and any cell matching more than one, or no, combination is left unannotated (labelled "Other"). Negative markers (`-`) are as important as positive ones: they exclude cells that share a positive marker with another cell type.


# Tutorial
The files in `tutorial/` walk through the full workflow on the sample data (`sample_data/QuantCellPaperO3`, one bone marrow section, 42,434 cells, 25 markers), in this order:

0. `00_QuPath_guide.md`: segment cells, call each marker positive or negative, and export one .csv per marker in QuPath.
1. `01_define_marker_combinations.ipynb`: write the marker combinations (positive and negative markers for each cell type) to a .json file.
2. Build the QuantCell input table `codex_rule_based_<project>.csv` with binary marker rule-based annotation, using either
   - `02a_initialization_qupath.ipynb`, from QuPath .csv exports, or
   - `02b_initialization_custom_table.ipynb`, from any one-row-per-cell table with feature columns and positive/negative marker calls.
3. `03_model_selection.ipynb`: compare candidate classifiers by cross-validated macro average precision on the training split, and optionally tune their hyperparameters on your data.
4. `04_annotation.ipynb`: train the selected model (LightGBM by default), annotate unannotated cells at a user-defined false discovery rate (FDR), check the result with PCA plots, and save `codex_quantcell_<project>.csv`.

`hyperparameters/` contains the tuned hyperparameters for all ten classifiers from the bone marrow analysis in the paper. They are a starting point; the best model and settings can differ between datasets.

`sample_data/expected_output.zip` contains the expected output of the tutorials on the sample data.

# Reproducing the paper
`paper/` contains the code for every analysis, figure and table in the paper, run from the Zenodo deposit and the public external datasets. See `paper/README.md` for the data layout, environments, run order and a figure-by-figure map. Figures are pixel-identical to the paper's with Matplotlib 3.7.1.

Data from the paper (raw PhenoCycler image, QuPath exports, and QuantCell input and output tables) are available on Zenodo: https://doi.org/10.5281/zenodo.23672666.

Video walkthroughs of the tutorial:
* Step 0, QuPath (segmentation, marker calling and export): https://youtu.be/HE_BD0LKfBY
* Steps 1-4, QuantCell notebooks: https://youtu.be/wcS-BahquwI

# License
   Copyright 2025 Rong Lu Lab

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

