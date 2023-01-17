# Understanding Neurons

## Setup

1. Clone the repository using recurse submodules, move to the repository root
   ```bash
   git clone --recurse-submodules https://github.com/zitkat/understanding-neurons.git
   ```
2. Simlink the data folder: 
   ```bash
    ln -s /storage/plzen4-ntis/projects/cv/understanding-critical-neurons-data data
   ```
3. You're good to go!

## Enviroment
Ready-to-go singularity image is in data folder, alternatively the corresponding defintion file is `lucent_torch_21.03-py3.def`. 
Use `requirements.txt` only when setting up python virtual enviroment. 

## TODO in infrastructure and visualization
- split into tools library and experiments repo:

Library:
- main folder with: attribution, feature rendering, criticallity, 
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