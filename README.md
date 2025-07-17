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

## Prerequisites for Replicating Paper Figures
* All of the prerequisites mentioned above
* SciPy 1.11.0
* seaborn 0.13.2
* pyyaml 6.0
* astir 0.1.2
* MAPS 1.0.0
* AnnoSpat 1.0.0

*QuantCell was tested using a Python environment with the prerequisite packages above on a dedicated server with Ubuntu 20.04.6 LTS. There are no other hardware or software dependencies but the authors make no guarantee it will work on every system.

To install the required packages into a new conda environment, you can use conda and pip:
(expected time: 1-3 minutes)
```
conda create -n "quantcell" python=3.11
conda activate quantcell
pip install numpy==1.26.0 pandas==1.5.3 scikit-learn==1.5.0 matplotlib==3.7.1 glob2==0.7 joblib==1.4.2 tqdm==4.65.0
pip install scipy==1.11.0 seaborn==0.13.2 pyyaml==6.0
```
## Running MAPS, astir, and AnnoSpat to create comparison data
QuantCell benchmarks itself against these three packages in the paper using the same dataset. The code used to benchmark them is provided in maps.ipynb, astir.ipynb, and annospat.sh. Due to version incompatibilities, these files will need to be run in separate python environments. Please see the associated GitHubs for instructions on how to install them:
* [MAPS](https://github.com/mahmoodlab/MAPS/)
* [astir](https://github.com/camlab-bioml/astir/)
* [AnnoSpat](https://github.com/faryabiLab/AnnoSpat/)


# Input
QuantCell requires two primary types of input data:
1. 

# Tutorial
Use initialization.ipynb to walkthrough the initialization of the .csv marker files, conventional annotation, and create a combined dataframe which is the input to QuantCell.

Use machine_learning.ipynb to walkthrough the annotation of cells using QuantCell with a Random Forest model with hyperparameters selected from the analysis in the paper.

A test dataset is provided along with the expected output.

Code to create all figures in the paper can be found in figure_data_generation.ipynb, paper_figures_formatted.ipynb, maps.ipynb, astir.ipynb, and annospat.sh.

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

