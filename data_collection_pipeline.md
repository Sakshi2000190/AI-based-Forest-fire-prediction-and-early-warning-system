 # Data Collection Pipeline — Uttarakhand Forest Fire Risk Prediction

This document explains what `00_boundary_extraction.py` and
`01_data_collection_pipeline.py` do, in order, and what each one produces.
It exists so teammates can understand and re-run the pipeline without
reading through every code comment.

## Overview

The project has two models:

1. **Static baseline model** (Random Forest / XGBoost) — one row per grid
   cell, using mean/composite values of weather, vegetation, and terrain.
2. **CNN-LSTM time-series model** — the main early-warning model. Each
   training row is a 4-week sequence of weather + vegetation for one grid
   cell, used to predict fire risk one week ahead.

Both models share the same underlying raw data (fire records, weather,
vegetation, terrain), collected once and processed into two different
shapes.

## Prerequisites

- Python packages: `earthengine-api`, `geemap`, `geopandas`, `rasterio`,
  `pandas`, `numpy`, `matplotlib`
- A registered, authenticated Google Earth Engine (GEE) account, with a GEE
  project ID (set as `FFP_GEE_PROJECT`)
- A GADM India state-boundary shapefile (`IND_adm1.shp`), used only to cut
  out Uttarakhand
- A FIRMS (NASA Fire Information for Resource Management System) fire
  detection CSV for Uttarakhand, 2015–2024, combining MODIS and VIIRS
  sensors

Before running either script, set `FFP_BASE_DIR` (and `FFP_GEE_PROJECT`,
`FFP_GADM_SHAPEFILE`, `FFP_BOUNDARY_OUTPUT` if needed) as environment
variables, or edit the CONFIG section at the top of each script.

## Step 0 — Boundary extraction (`00_boundary_extraction.py`)

GADM's national shapefile covers every Indian state, labeled by its
pre-2007 name `Uttaranchal` rather than `Uttarakhand`. This script filters
that one state out and saves it as its own shapefile
(`gis_data/raw/uttarakhand_boundary.shp`), which every later export uses to
clip data down to just Uttarakhand.

## Step 1 — Data collection pipeline (`01_data_collection_pipeline.py`)

### 1. Setup
Connects to Earth Engine, loads the Uttarakhand boundary, and confirms its
area (~53,483 sq km) as a sanity check. Must be re-run at the start of
every new kernel session — GEE authentication and loaded objects don't
persist.

### 2. Fire occurrence data (FIRMS)
Loads the FIRMS fire CSV, filters it to 2015–2024, clips it to the
Uttarakhand boundary, and saves the result as
`gis_data/processed/fire_uttarakhand_clipped.csv`. This is the label data
for both models. A map of fire points is also saved to
`gis_data/outputs/fire_points_map.png`.

### 3. Baseline exports (vegetation, weather, topography)
Three one-time exports to Google Drive, each a single composite/mean image
covering the full 2015–2024 period:

| Dataset | Source | Scale | Purpose |
|---|---|---|---|
| NDVI / NDMI | Sentinel-2 (median composite) | 30m | Vegetation health & moisture |
| Weather | ERA5-Land (mean composite, 8 variables) | 1km | Temperature, humidity, wind, rainfall, pressure, soil moisture |
| Elevation / slope / aspect | SRTM | 30m | Static terrain |

These have already been run and downloaded, so the export code is commented
out — re-running it would queue duplicate GEE exports and burn quota. What
runs is a verification step that opens the downloaded `.tif` files and
prints their band count and shape.

### 4. Static grid-features dataset (baseline model input)
All three baseline rasters are reprojected onto a common reference grid
(the weather raster's grid), fire points are rasterized onto the same grid
as a binary label, and everything is flattened into one row-per-cell
dataframe. Cells outside the state boundary (NaN) and invalid elevation
values are dropped.

**Output:** `gis_data/processed/uttarakhand_grid_features.csv` — roughly
60,000 rows, columns: `longitude`, `latitude`, `NDVI`, `NDMI`, 8 weather
variables, `elevation`, `slope`, `aspect`, `fire_occurred` (0/1).

### 5. Time-series design decisions
Before building the CNN-LSTM dataset, four design choices were finalized:

- **Weekly resolution** — daily would mean 3,650+ time steps over 10 years
  and worse cloud-cover gaps in vegetation data; weekly still captures
  fire-driving weather trends (dryness, heat build-up).
- **4-week lookback** — enough to show a dryness/temperature trend without
  adding too much noise or complexity.
- **1-week-ahead prediction** — matches the practical use case of an
  early-warning system.
- **2km grid** — the original 1km grid produced a 13GB CSV; 2km keeps the
  dataset a manageable size while still being fine enough for early
  warning.

### 6. Weekly weather export (2km)
ERA5 weather, re-exported as **weekly** means (rather than a single mean),
split by year, at 2km scale. Already run and downloaded to
`gis_data/gee_exports/weekly_weather_2km/`; the export code is commented
out, followed by a verification step.

### 7. Land cover export
A static ESA WorldCover 2021 layer, exported directly at 2km scale (10
land-cover classes: tree cover, shrubland, grassland, cropland, built-up,
etc.). Used as an extra static feature. Already run; resampled onto the
reference grid using nearest-neighbor (since the values are categorical
class codes, not continuous data).

### 8. Monthly vegetation export
NDVI/NDMI, re-exported as **monthly** composites (weekly wasn't practical —
monsoon cloud cover leaves large gaps), split by year, at 250m scale
(reduced from an initial 30m attempt that produced oversized, sharded
files and exceeded GEE's non-commercial quota). Already run.

### 9. Forward-fill missing vegetation months
Some months have no valid Sentinel-2 image (monsoon cloud cover), and 2015
only has October–November data since Sentinel-2 launched in June 2015. For
each pixel, missing months are filled with the last valid NDVI/NDMI
reading rather than left blank.

### 10. Weekly fire labels
Fire records are grouped into the same weekly bins used for the weather
data, then rasterized into one binary grid per week (fire / no fire).

### 11. Building the final time-series dataset
For each week (once at least 4 weeks of history exist), a training row is
built per valid land pixel:

- past 4 weeks of all 8 weather variables (`{variable}_lag0..lag3`)
- current month's NDVI/NDMI (forward-filled)
- static elevation, slope, aspect, land cover
- **label:** `fire_next_week` — whether a fire was detected in that cell
  the following week

Rows are written to disk incrementally, one week at a time, since the full
dataset doesn't fit in memory.

**Output:** `gis_data/processed/uttarakhand_timeseries_features_2km.csv`

### 12. Cleaning
Rows where NDVI/NDMI are still missing after forward-fill (edge cases with
no valid vegetation data at all in the lookback window) are dropped, in
500k-row chunks to keep memory use low.

**Output:** `gis_data/processed/uttarakhand_timeseries_features_2km_clean.csv`
— ~6.9 million rows, 42 columns. The `fire_next_week` label is highly
imbalanced (~0.9% positive), which the CNN-LSTM training step will need to
account for (e.g. class weighting, PR-AUC as the evaluation metric rather
than accuracy).

### 13. Packaging for sharing
The clean CSV is zipped (`uttarakhand_timeseries_features_2km_clean.zip`)
for handing off to the teammate building the CNN-LSTM model.

## Output summary

| File | Rows | Used for |
|---|---|---|
| `fire_uttarakhand_clipped.csv` | — | Shared label source |
| `uttarakhand_grid_features.csv` | ~60,000 | Baseline RF/XGBoost model |
| `uttarakhand_timeseries_features_2km_clean.csv` | ~6.9M | CNN-LSTM model |

## Notes for teammates

- None of the large data files (`.tif`, `.csv`, `.zip`, `.shp`) are
  committed to Git — they're in `.gitignore`. Get them from the shared
  Google Drive folder instead.
- Set `FFP_BASE_DIR` and `FFP_GEE_PROJECT` as environment variables (or
  edit the CONFIG section) before running either script — hardcoded local
  paths won't work on your machine.
- The commented-out GEE export blocks are intentional — they document what
  was already run rather than being incomplete code. Don't uncomment and
  re-run them; it would queue duplicate exports and use up GEE's
  non-commercial quota again.
