# Fixed Evaluation Locations

**Date:** 2026-07-08 
**Dataset:** Sen1Floods11 v1.1

---

## 1. Purpose

Five geographic locations are fixed as evaluation sites before model training begins. This prevents selection bias, ensures reproducible results across experiments, and provides coverage of the flood environments most relevant to the project's deployment target — Azerbaijan.

The set combines **two Azerbaijan sites** (evaluated against independently acquired Sentinel-1 imagery) with **three confirmed Sen1Floods11 dataset locations** (evaluated against expert-labeled chips).

---

## 2. Dataset Analysis Summary

**Sen1Floods11**

| Property | Value |
|---|---|
| Flood events | 11 distinct events |
| Countries | Bolivia, Colombia, Ghana, India, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri Lanka, USA |
| Total chips | 4,831 non-overlapping 512 × 512 px tiles |
| Hand-labeled chips | 446 (expert ground truth, used for evaluation) |
| Weakly-labeled chips | ~4,385 (algorithmic labels, for training only) |
| Resolution | 10 m (Sentinel-1 VV + VH, EPSG:4326) |
| Temporal range | 2016–2019 |

**Key characteristics:**

- Covers 6 continents, 14 biomes, 357 ecoregions
- Flood types: monsoon riverine, seasonal tropical, savanna pluvial, Mediterranean flash, semi-arid riverine, temperate snowmelt
- Terrain: alluvial plains, tropical forest, savanna, coastal lowland, Mediterranean foothills
- Predominantly rural; dense urban flood scenes are underrepresented
- **Azerbaijan is not included.** No event covers the Caucasus, Central Asia, or any adjacent region.

---

## 3. Selected Locations

### Baku, Azerbaijan *(not in Sen1Floods11)*

| Attribute | Value |
|---|---|
| Country | Azerbaijan |
| Flood type | Urban flash flooding; Caspian Sea coastal inundation |
| Terrain | Dense urban; Absheron Peninsula; low-lying coastal zone |
| Climate | Semi-arid (BSk); ~200–300 mm/yr |

**Why selected:**
- Fixed by project requirement as the primary deployment site.
- Represents Azerbaijan's urban and critical-infrastructure flood risk (port, oil terminals, government).
- Provides a challenging urban environment where buildings can make flood detection with Sentinel-1 SAR more difficult.
- Caspian Sea storm surges and aging drainage infrastructure create compound flood hazards.

---

### Sabirabad, Azerbaijan *(not in Sen1Floods11)*

| Attribute | Value |
|---|---|
| Country | Azerbaijan |
| Location | Kura–Aras river confluence, Central-Southern Azerbaijan |
| Flood type | Large-scale riverine; levee overtopping; prolonged agricultural inundation |
| Terrain | Flat alluvial floodplain; irrigated cropland; earthen levees |
| Climate | Semi-arid (BSk/BWk); spring snowmelt and rainfall driven |

**Why selected:**
- Highest flood impact of any district in Azerbaijan: ~43,726 ha inundated during the 2010 Kura flood (worst in over a century), plus major events in 2003 and 2007.
- Located at the confluence of the Kura and Aras rivers, making it highly vulnerable to flooding.
- Flat terrain and sparse vegetation allow clear flood detection with Sentinel-1 SAR.
- Represents the riverine flooding conditions that are a primary focus of this project.

---

### DS-1 · Assam / Brahmaputra Valley, India 

| Attribute | Value |
|---|---|
| Country | India |
| Region | Assam state; Brahmaputra and Barak river floodplains |
| Flood type | Monsoon riverine; large-scale seasonal inundation on a braided river system |
| Terrain | Broad alluvial floodplain; braided channels; paddy agriculture |
| Climate | Tropical monsoon (Am); June–September monsoon |
| In Sen1Floods11 | Event ID 3; S1 acquired 2016-08-12; 535 chips (467 train + 68 val) |

**Why selected:**
- Contains the largest number of hand-labeled samples in the dataset, providing reliable evaluation results.
- Represents large-scale monsoon river flooding, one of the most common flood types worldwide.
- Includes dense vegetation and complex river channels, making flood detection more challenging.
- Has river and floodplain characteristics similar to the Kura River, making it a useful reference for Azerbaijan.

---

### DS-2 · Indus River Floodplain, Pakistan 

| Attribute | Value |
|---|---|
| Country | Pakistan |
| Region | Indus River floodplain; Sindh or Punjab province |
| Flood type | Large-scale riverine; monsoon-amplified alluvial plain inundation |
| Terrain | Flat irrigated alluvial plain; minimal topographic relief |
| Climate | Semi-arid to sub-humid (BSk); seasonal monsoon influence |
| In Sen1Floods11 | Confirmed event; exact chip count in GeoJSON metadata |

**Why selected:**
- Has environmental conditions similar to Sabirabad, including a semi-arid climate, flat floodplains, and a mountain-fed river.
- Represents flood conditions that are relevant to the project's deployment in Azerbaijan.
- Flat terrain and sparse vegetation provide clear Sentinel-1 SAR flood observations.

**Limitations:** Irrigated fields may be mistaken for flooded areas. The exact image acquisition date should be verified in Sen1Floods11_Metadata.geojson

---

### DS-3 · Llanos de Mojos, Bolivia 

| Attribute | Value |
|---|---|
| Country | Bolivia |
| Region | Llanos de Mojos, Beni Department (Moxos Province / Trinidad Municipality) |
| Flood type | Seasonal Amazon tributary inundation; tropical forest flood pulse |
| Terrain | Dense tropical forest + seasonally flooded savanna; very flat Amazon lowland |
| Climate | Tropical humid (Af/Am); austral summer rainy season (December–April) |
| In Sen1Floods11 | Event ID 1; S1 acquired 2018-02-15; 239 chips (224 train + 15 val) |

**Location note:** The GeoJSON `location` field records only `"Bolivia"`. The specific region is identified by cross-referencing the event bounding box (11.4°S–16.0°S, 64.4°W–65.6°W) with UNOSAT satellite analysis of the same February 2018 event, which confirmed Moxos Province (>35,500 ha) and Trinidad Municipality (~6,900 ha) as the primary affected areas along the Mamoré River.

**Why selected over Colombia (534 train, 0 val chips) and Paraguay (incomplete metadata):**
- Colombia has zero validation chips and cannot contribute evaluation metrics.
- Bolivia is the only South American event with confirmed hand-labeled validation chips.

**Why selected for this evaluation set:**
- Represents the only tropical forest flooding scenario in the selected locations, making flood detection more challenging.
- Includes seasonal Amazon floodplain flooding, which is different from the other selected flood types.
- Adds South America, increasing the geographic and environmental diversity of the evaluation.
- Commonly used in Sen1Floods11 studies, allowing easier comparison with published results.

**Limitations:** Contains only 15 validation samples, so evaluation results are less reliable. Dense forest also makes flood detection with Sentinel-1 SAR more challenging, so lower performance is expected.

---

## 4. Comparison Table

| ID | Location | Country | In Sen1Floods11 | Flood Type | Terrain | Climate | Urban/Rural |
|---|---|---|---|---|---|---|---|
| AZ-1 | Baku | Azerbaijan | No | Urban flash; Caspian coastal | Dense urban; coastal | Semi-arid (BSk) | Highly urban |
| AZ-2 | Sabirabad | Azerbaijan | No | Riverine; levee overtopping | Flat alluvial; cropland | Semi-arid (BSk/BWk) | Rural |
| DS-1 | Assam / Brahmaputra Valley | India | Yes (ID 3) | Monsoon riverine | Alluvial; braided rivers; paddy | Tropical monsoon (Am) | Rural / peri-urban |
| DS-2 | Indus River Floodplain | Pakistan | Yes | Semi-arid alluvial riverine | Flat irrigated plain | Semi-arid (BSk) | Rural |
| DS-3 | Llanos de Mojos, Beni Dept. | Bolivia | Yes (ID 1) | Amazon seasonal inundation | Tropical forest + flooded savanna | Tropical humid (Af/Am) | Remote / rural |

### Diversity at a Glance

| Dimension | Coverage |
|---|---|
| Continents | South Caucasus, South Asia, South America |
| Climate zones | Semi-arid · Tropical monsoon · Tropical humid |
| Flood mechanisms | Urban flash · Riverine confluence · Monsoon riverine · Semi-arid alluvial · Amazon forest pulse |
| Terrain classes | Urban · Alluvial plain · Braided floodplain · Irrigated plain · Tropical forest |
| SAR difficulty | Urban-complex (Baku) · Easy-moderate (Sabirabad, Pakistan) · Moderate-hard (India) · Hardest (Bolivia) |
| Label source | Expert chips: India, Pakistan, Bolivia · Independent S1 archive: Baku, Sabirabad |

---

## 5. DEM-Derived Layover/Shadow Masking (G24)

**Date:** 2026-07-12

A DEM-derived local-incidence-angle mask now flags Sentinel-1 pixels affected by
radar layover or shadow before they reach the model or are scored
(`src/data/preprocessing.py::layover_shadow_mask`). Terrain slope/aspect is
computed from a Copernicus DEM mosaic reprojected onto the scene grid; the local
incidence angle is derived from slope, aspect, and the sensor's look geometry
(mid-swath IW incidence angle 29.1°–46.0°, look azimuth from `orbit_pass`).
Layover is flagged where local incidence ≤ 0°, shadow where it is ≥ 90°.

**DEM source and a real coverage gap.** DEM tiles are pulled from the public
Copernicus DEM AWS bucket (`scripts/download_dem.py`), preferring GLO-30 (30 m)
and falling back to GLO-90 (90 m) per 1°×1° tile. While building this feature we
found that **GLO-30 has no coverage over Azerbaijan**: tiles
`Copernicus_DSM_COG_10_N40_00_E049_00_DEM` (Baku) and
`Copernicus_DSM_COG_10_N40_00_E048_00_DEM` (Sabirabad) both 404 on the GLO-30
bucket, while neighboring tiles at the same latitude (e.g. `E042`, `E051`) exist.
GLO-90 covers both, so both fixed Azerbaijan sites currently run on 90 m terrain
instead of 30 m — coarser slope estimates than the rest of the fixed set.

**Effect on the two Azerbaijan sites.** Within the fixed 512×512 evaluation
footprints, the mask flags **0% of pixels** at both sites:

| Site | DEM resolution | Elevation range | Max slope | Layover/shadow |
|---|---|---|---|---|
| Baku | GLO-90 (fallback) | 8.8–66.5 m | ~10.7° | 0.00% |
| Sabirabad | GLO-90 (fallback) | -16.5–-8.6 m | ~6.3° | 0.00% |

Both footprints are low-relief (urban-coastal lowland and river floodplain,
respectively) and stay well inside the ~29–46° incidence window, so neither
currently exercises the mask. This is not evidence the mask is inert: the
surrounding Absheron/Caucasus foothills, within the same 1°×1° DEM tile as
Baku, reach slopes up to ~38.5° and elevations over 1,500 m — enough to trigger
both layover and shadow. If the Baku AOI is ever widened toward those hills,
this mask should be re-checked, and a GLO-30 source (or a licensed substitute)
revisited if finer terrain detail becomes necessary.

**Known approximations.** No per-scene orbit state vectors are available for
these exports (`scripts/pull_s1_scene.py` does not capture
`orbitProperties_pass`), so `orbit_pass` is a config-level assumption
(`ASCENDING` by default) rather than a per-scene fact, and the incidence angle
uses the fixed IW mid-swath value rather than a true per-pixel range-dependent
angle. Both are documented, overridable knobs
(`config/default.yaml: inference.layover_shadow`), not hidden constants.

## 6. References

1. Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). *Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1.* CVPR Workshops 2020.
   http://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html

2. Cloud to Street. *Sen1Floods11 GitHub Repository.*
   https://github.com/cloudtostreet/Sen1Floods11

3. Cloud to Street. *Sen1Floods11_Metadata.geojson* (authoritative event metadata).
   https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/Sen1Floods11_Metadata.geojson

4. Google Cloud Storage bucket: `gs://sen1floods11`

5. UNOSAT. *Satellite Detected Water in Beni Department, Bolivia — February 2018.*
   https://reliefweb.int (UNOSAT analysis, RADARSAT-2, 2018-02-12)

6. Asian Disaster Reduction Center (ADRC). *Azerbaijan Kura River Flood, 2010.*
   https://www.adrc.asia

7. UNECE. *Transboundary Flood Risk Management in the Kura–Aras Basin.*
   https://unece.org

8. ESA. *Sentinel-1 SAR Technical Guide.*
   https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-1-sar
