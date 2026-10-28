/**
 * Export the cell measurements for one marker to a .csv file. Same result as Measure > Export measurements with
 * only that marker's measurement columns selected, without ticking the columns one by one.
 *
 * Run it after segmenting the cells and applying the marker's object classifier, once per marker.
 * It asks for the channel name as it appears in the measurements (for example CD45) and where to save the file.
 */
import qupath.lib.gui.dialogs.Dialogs

def marker = Dialogs.showInputDialog('Export marker measurements', 'Channel name as it appears in the measurements (e.g. CD45)', '')
if (!marker)
    return
def file = Dialogs.promptToSaveFile('Save as', null, marker + '.csv', 'CSV', '.csv')
if (file == null)
    return

def cells = getCellObjects()
if (!cells) {
    println 'No cells found'
    return
}
def names = cells[0].getMeasurementList().getMeasurementNames()
def markerCols = names.findAll { it.startsWith(marker + ':') }
if (!markerCols) {
    println "No measurements start with '${marker}:'. Channels found: " + names.findAll { it.contains(':') }.collect { it.split(':')[0] }.unique()
    return
}

def fixed = ['Image', 'Name', 'Class', 'Parent', 'ROI', 'Centroid X µm', 'Centroid Y µm']
def shape = names.findAll { !it.contains(':') }
saveDetectionMeasurements(file.getAbsolutePath(), *(fixed + shape + markerCols))
file.text = file.text.replace('\t', ',')   // the export uses the table delimiter from Preferences (tab by default)
println "Saved ${file} (${markerCols.size()} ${marker} columns)"
