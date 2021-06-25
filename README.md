# Understanding Critical Neurons

## Setup

1. Clone the repository, move to the repository root
2. Simlink the data folder: 
   ```bash
    ln -s /storage/plzen4-ntis/projects/cv/understanding-critical-neurons-data data
   ```
3. You're good to go!

## Enviroment
Ready-to-go singularity image is in data folder. Use `requirements.txt` only when setting up
python virtual enviroment.

## TODO in infrastructure and visualization
- effectively render multiple neurons from different layers
- different neuron sampling strategies 
- incorporate pruning and switching off neurons into mapped model (with caching)
- circuits backprop
- max activation search
- add GRADCam
- Jupyter notebook with example visualizations