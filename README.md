# OS Map
Ordnance Survey Terrain 50 (https://osdatahub.os.uk/data/downloads/open/Terrain50) provides elevation data for the entirety of the UK

## Objectives
* Read OS Terrain 50 data (GML grid and contours) using Python
* Apply boundaries to data to allow defined areas to be analysed
* Plot data to produce visualisations of the elevation data

## Data Format
Data is provided for the UK in 10km x 10km tiles with data points at 50m intervals

### Tile Structure and Coverage
* Tile Naming: Tiles are identified by the Ordnance Survey National Grid reference of their southwest corner (e.g., TL or SU blocks).
* Grid Specifications: The gridded Digital Terrain Model (DTM) has a regular post spacing of 50 metres, structured as a 200 x 200 matrix of elevation values per tile.
* Coordinates: Coordinates mark the centre of each cell rather than its corner, ordered in row-major sequence (x easting within y northing).

### Data Formats
* Grid Data: Provided as text-based ASCII grid files paired with GML metadata files defining the spatial reference system and cell spacing.
* Contour Data: Supplied as vector datasets featuring 10-metre vertical interval standard contour polylines, spot heights, and mean high/low water boundaries.

### ASCII Grid and GML (Grid)
* The downloaded file contains a zip file which contains individual zip files for 

## Notes
To recursively unzip the downloaded files, use the following 
find . -name "*.zip" -exec unzip {} \;
This unzips recursive folders in the working directory to the working directory
For Grid data, it is the .ASC files we care about (everything else can be deleted)
