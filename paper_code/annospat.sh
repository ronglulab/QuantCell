#!/bin/bash

main_folder="/store/Projects/wboohar/PhenoCycler/QuantCellPaper/AnnoSpat"

start=`date +%s%N`
mkdir "$main_folder/outputdir"
AnnoSpat generateLabels -i "$main_folder/marker_data.csv" \
-m "$main_folder/marker_combos.csv" \
-o "$main_folder/outputdir" \
-f B220 -l Ter119 -r ROI -t "[99.9,99.999,70]" -a "[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99]"
end=`date +%s%N`
runtime=$((end-start))
rm "$main_folder/time_elapsed.txt"
touch "$main_folder/time_elapsed.txt"
echo "$runtime" >> "$main_folder/time_elapsed.txt"
