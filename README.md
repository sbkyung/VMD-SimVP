# VMD-SimVP

This repository provides example code accompanying the manuscript:

**Resolving multiscale drought information improves subseasonal prediction of integrated drought conditions**

Bokyung Son, Sihun Jung, Jaese Lee, Jungho Im, and Sumin Park

**Manuscript status:** Under revision.


## Important Notes

The current version of this repository provides example implementations associated with the manuscript. Additional materials, including example data and expanded documentation, will be added following publication.


## Overview

The repository illustrates two components of the proposed drought prediction framework:

1. **Sequential variational mode decomposition (VMD)**  
   Example code for performing sequential VMD using an author-modified version of the `vmdpy` implementation, provided in the `vmd` directory.

2. **SimVP-based drought prediction**  
   Example code illustrating the SimVP model architecture adapted for predicting United States Drought Monitor (USDM) conditions and their decomposed components.

Detailed descriptions and usage instructions will be expanded following publication.


## Citation

The manuscript is currently under revision. The full citation and DOI will be added once the paper is published.

If you use the underlying VMD or SimVP implementations, please also cite the relevant original work listed below.

## Original Implementations and References

This work builds on the following publicly available implementations.

### vmdpy

The VMD implementation is adapted from the Python `vmdpy` package.

- Original repository: [vrcarva/vmdpy](https://github.com/vrcarva/vmdpy)
- Reference: Carvalho, V. R., Moraes, M. F. D., Braga, A. P., and Mendes, E. M. A. M. (2020). Evaluating five different adaptive decomposition methods for EEG signal seizure detection and classification. *Biomedical Signal Processing and Control*, **62**, 102073. https://doi.org/10.1016/j.bspc.2020.102073

### SimVP

The drought prediction models are adapted from the original SimVP implementation.

- Original repository: [A4Bio/SimVP](https://github.com/A4Bio/SimVP)
- Reference: Gao, Z., Tan, C., Wu, L., and Li, S. Z. (2022). SimVP: Simpler Yet Better Video Prediction. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 3170–3180.


## Contact

For questions, please contact:

**Bokyung Son**  
[sbkyung0@unist.ac.kr](mailto:sbkyung0@unist.ac.kr)
