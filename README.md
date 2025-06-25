# Understanding Neurons

## Setup

Clone the repository using recurse submodules, move to the repository root
   ```bash
   git clone --recurse-submodules https://github.com/zitkat/understanding-neurons.git
   ```

## Enviroment
Ready-to-go singularity image is in data folder, alternatively the corresponding defintion file is `lucent_torch_21.03-py3.def`. 
Use `requirements.txt` only when setting up python virtual environment. 

## TODO in infrastructure and visualization
- split into tools library and experiments repo:

Library:
- main folder with: attribution, feature rendering, criticality, 
- effectively render multiple neurons from different layers
- different neuron sampling strategies 
- incorporate pruning and switching off neurons into mapped model (with caching)
- circuits backprop
- max activation search
- add GRADCam
- Jupyter notebook with example visualizations

Experiments:
- datasets
- rendering scripts, settings?
- visualizations