# Mixed-data decision

The detector remains a one-class `bird` model. A generic detection class permits combining valid bird boxes from several datasets without inventing species-level labels.

Open Images is capped at 20,000 `Bird`-boxed images. Its complete Ultralytics-formatted release is documented as 561 GB and must not be downloaded: it would leave too little working space for processing, checkpoints, and results on the available D: drive.

iNaturalist 2021 is intentionally excluded. It supplies image-level species labels, not bounding boxes, so it cannot be converted into a trustworthy detector training set without new annotation work.

NABirds and CUB data can only support the research baseline. Their terms restrict use, so no model trained on them may be treated as cleared for a commercial or operational deployment without separate rights review.

Before combining sources, the preparation workflow must compare file hashes and perceptual hashes, then split by source image before train/validation/test assignment. Exact duplicates and near-identical source images must not cross splits.
