# From image to QuantCell input with QuPath

This guide covers the steps before the notebooks: segmenting cells, calling each marker positive or negative, and exporting
one .csv file per marker. The paper used QuPath 0.3.0. QuPath is not required: if your segmentation and marker calls come
from another tool, skip to `02b_initialization_custom_table.ipynb`.

QuPath's own documentation for these steps:
- Multiplexed analysis (cell classification per marker): https://qupath.readthedocs.io/en/0.4/docs/tutorials/multiplex_analysis.html
- Exporting measurements: https://qupath.readthedocs.io/en/0.4/docs/tutorials/exporting_measurements.html

## 1. Open the image
Create a QuPath project and add the multiplexed image (for PhenoCycler data, the .qptiff). Check that the channel names and
the pixel size (Image tab) are correct, because measurements are reported in µm.

## 2. Segment cells
Draw or select the tissue region(s) to analyse, then run the StarDist script in this repository
(`StarDist cell segmentation script_0.37_um_per_pixel.groovy`, Automate > Show script editor). It asks for a StarDist
nucleus model file (.pb), segments nuclei on the DAPI channel (channel 0) and approximates each cell by expanding the
nucleus by up to 5 µm (limited to 1.5x the nucleus size). It also measures every channel in each compartment (nucleus,
cytoplasm, membrane, cell) with the mean, median, minimum, maximum and standard deviation. Set `.pixelSize()` to your
image's resolution if it is not 0.37 µm.

## 3. Call each marker positive or negative
Repeat for every marker used in your marker combinations:
1. Create a class for the marker (for example `CD45`). Its name does not have to match the marker combinations exactly;
   QuantCell identifies the marker from the file name in step 4.
2. Annotate a few dozen clearly positive cells with that class and a few dozen clearly negative cells as `Other`, spread
   across the tissue (the paper used 20 to 70 of each per marker).
3. Train an object classifier on that marker's measurements (Classify > Object classification > Train object classifier)
   and apply it to all cells.
4. Review the calls on the image. If cells are misclassified, add training cells and retrain until the calls look right.
   Treat a cell as negative when the apparent signal comes from a neighbouring positive cell.

## 4. Export one .csv per marker
For each marker, export the detection measurements (Measure > Export measurements, object type: cells) after applying
that marker's classifier, and save the file as `<marker>.csv` (for example `CD45.csv`). Put all marker files for one image
or sample in one folder. QuantCell expects:
- one file per marker in the marker combinations, named after the marker (`<marker>.csv` or `<marker> (<label>).csv`);
- every file exported from the same set of detected cells, in the same order (QuantCell checks that the centroids match);
- the call in the `Name`/`Class` column: the marker's class for positive cells, `Other` for negative cells (cells left
  unclassified, `PathCellObject`, are treated as unknown and cannot match a combination that uses that marker);
- the standard QuPath columns (`Image`, `Name`, `Class`, `Parent`, `ROI`, `Centroid X µm`, `Centroid Y µm`, ...), and
  the marker channel's measurement columns, `<channel>: <Nucleus|Cytoplasm|Membrane|Cell>: <Mean|Median|Min|Max|Std.Dev.>`.
  The channel name may differ slightly from the marker name (for example `CD117 (Cy5)` or `F4/80` for `F4_80`); QuantCell
  matches them and renames the columns to the marker name.

Instead of ticking the columns in the dialog, you can run `qupath_marker_export.groovy` from this repository
(Automate > Show script editor): it asks for the channel name and where to save, and exports the same columns as above
for that marker. Leaving the dialog's column selection empty also works; the file then contains every channel's
measurements, which QuantCell ignores.

`sample_data/QuantCellPaperO3` is an example of a correct export (one bone marrow section, 25 markers).

## 5. Next
Place each sample's folder of marker files inside one data folder and continue with
`02a_initialization_qupath.ipynb`.
