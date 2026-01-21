# Cross-view Domain Generalization via Geometric Consistency for LiDAR Semantic Segmentation

The official implementation for [Paper Link]()

## Abstract
Domain-generalized LiDAR semantic segmentation seeks to learn models from source-domain data that generalize reliably to multiple unseen target domains, which is essential for real-world LiDAR applications. However, existing approaches assume similar acquisition views (e.g., vehicle-mounted) and struggle in cross-view scenarios, where observations differ substantially due to viewpoint-dependent structural incompleteness and non-uniform point density. Accordingly, we formulate cross-view domain generalization for LiDAR semantic segmentation and propose a novel framework, termed CVGC (Cross-View Geometric Consistency). Specifically, we introduce a cross-view geometric augmentation module that models viewpoint-induced variations in visibility and sampling density, generating multiple cross-view observations of the same scene. Subsequently, a geometric consistency module enforces consistent semantic and occupancy predictions across geometrically augmented point clouds of the same scene. Extensive experiments on two benchmark suites demonstrate that CVGC consistently outperforms state-of-the-art methods when generalizing from a single source domain to multiple target domains with heterogeneous acquisition viewpoints.

# News
We will release our code soon.

# Usage

## Testing
```
python test.py
```
[Here]() is test weight for H3D to PL3D&ISPRS.
[Here]() is test weight for STPLS3D to T3D&DALES.


# Cite
Please cite our work if you find it useful.
```
Bibtex
```
