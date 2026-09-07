# Public image dataset investigation

Reviewed: 2026-08-09. “Cornell Bird Dataset” is interpreted here as the commonly referenced **Caltech-UCSD Birds 200-2011 (CUB-200-2011)** dataset; it is not the same dataset as NABirds. Confirm the intended dataset if a different Cornell release was meant.

| Candidate | Images / classes | Bird/crow coverage | Boxes and format | Detection suitability / practical assessment |
|---|---:|---|---|---|
| COCO 2017 | 118,287 train + 5,000 val images; 80 object categories | `bird` is one broad category; no crow category | Instance boxes and masks in COCO JSON | Excellent first general-bird detector. Mature tooling and a modest, manageable source release. |
| Open Images V7 | ~9.18M images; 1.74M train box images; 601 boxable classes | `Bird` has boxes; `Crow` is not a listed box class (related labels include Raven, Magpie) | Normalized coordinates in CSV | Technically excellent but the full detection release is about 561 GB and unsuitable as a first local download. A targeted `Bird` subset is a later option. |
| iNaturalist 2021 | 3.3M licensed photos, 10,000 species | Many bird species and likely crow taxa | Image-level taxon labels, not detection boxes | Classification data; not direct YOLO detection training without creating boxes. Per-image licenses must be preserved. |
| NABirds V1 | 48,562 images; 555 visual categories / 404 species with >=100 examples | North-American birds, including crow taxonomy | Per-image bounding boxes plus fine-grained labels/parts | Valid detector source and useful later for fine-grained/crow work, but image domain is not drone/pasture and terms restrict it to non-commercial research. |
| CUB-200-2011 | 11,788 images; 200 species | Includes American Crow in the class taxonomy | One bounding box per image, parts and attributes in text files | Valid, compact detection source but small and restrictive; better for research evaluation or supplementing than the initial broad detector. |

## Recommendation: initial YOLO data

Use **COCO 2017 with `class 0 = bird`** for the first model. It is an object-detection dataset with a single broad bird category, standard COCO JSON annotations, and a practical size. It does not claim crow-specific performance; that is intentional for a first detection baseline.

Do **not** download all of Open Images V7: the Ultralytics configuration documents a 561 GB download, which would consume most of the 676 GB currently free on D:. If general-bird performance needs more diversity after the COCO baseline, build a selective Open Images `Bird` subset and document its exact sample count and license metadata.

## Class design

### Option A — `0: bird` (selected)

Detects any bird-like target. It maximizes usable labeled boxes in COCO, enables a realistic detector baseline, and is appropriate until project-specific imagery establishes which species and viewpoints matter. It will also detect harmless/non-target birds, so downstream policy must not equate a detection with an intervention.

### Option B — `0: crow` (not selected yet)

Detects only an explicitly labeled crow class. Neither COCO 2017 nor Open Images V7 provides a `Crow` box class, so labeling their generic `bird` boxes as crow would be incorrect. NABirds/CUB have useful species labels but are too limited and license-constrained to support a robust deployment claim. A crow model needs a verified crow-only box dataset plus later, permissioned application-specific annotations.

## Licenses and use constraints

- **COCO:** images originate from Flickr; consult the official COCO terms/licensing for each intended use and retain required attribution. Annotations are distributed as part of COCO; do not assume commercial clearance simply because the data are public.
- **Open Images:** annotations are CC BY 4.0; source images are listed as CC BY 2.0, with an explicit warning to verify individual image licenses.
- **iNaturalist:** licenses are item-specific; default observation media are CC BY-NC and some are all-rights-reserved. Download only records whose individual licenses fit the intended use.
- **NABirds:** non-commercial research only; original data cannot be redistributed and derived data carry the same terms.
- **CUB-200-2011:** restricted to non-commercial research and educational use because the authors do not own the image copyrights.

License compatibility for a future deployed deterrence product requires a separate review; this document is technical guidance, not legal advice.

## Primary sources

- COCO detection: https://cocodataset.org/dataset/detection-2017.htm
- Open Images V7 facts, downloads and licenses: https://storage.googleapis.com/openimages/web/factsfigures_v7.html
- Open Images V7 format: https://storage.googleapis.com/openimages/web/download_v7.html
- iNaturalist developer datasets: https://www.inaturalist.org/pages/developers
- iNaturalist license policy: https://help.inaturalist.org/en/support/solutions/articles/151000175695
- NABirds terms: https://dl.allaboutbirds.org/merlin-computer-vision-terms-of-use
- CUB-200-2011: https://www.vision.caltech.edu/datasets/cub_200_2011/
