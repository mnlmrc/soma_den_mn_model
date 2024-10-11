# Soma-dendrite motor neuron model

## Overview 
Two compartment motor neuron (MN) pool model based on ([Elias and Kohn, 2013](https://www.sciencedirect.com/science/article/abs/pii/S0925231212006078?casa_token=T01r_7fUvHcAAAAA:ZbQ1gO_hS-TRkqEL70NnAckEG-5ZO7FmE9zGRCobI_8ZWbo9Iak_m2XaJK45fI0tbjg5lQVe)). Every MN is comprised by a cylindrical soma and dendrite. The dendrites of the MNs include L-type Ca2+ channels to simulate the effect of persistent inward currents. The output is coupled with a model of motor unit (MU) twitch for force production. The model support the injection of current in the soma or dendrite, plus synaptic excitatory inputs (generated based on [Avrillon et al, 2023](https://www.biorxiv.org/content/10.1101/2023.02.07.527433v1.abstract) and [Farina and Negro, 2015](https://journals.lww.com/acsm-essr/fulltext/2015/01000/Common_Synaptic_Input_to_Motor_Neurons,_Motor_Unit.6.aspx?casa_token=szN6TxNwHvUAAAAA:8S2rH0ZZkY0TrmBzmT2U4Bl3LAfpbBh-NPKPgMZxUIaQVPJi1RvWoUCEL3-Dcusb26mXQGcNU5tL2jMKcpQ3CxL9oA&casa_token=2VxzWnIaghYAAAAA:JMTlCRwgQ6ZDoTdhkBJvZ722bskOXZpnmpXBAWY6tWq0PmO9731auCVmkBHdtd2lAQxY_pdheSK3jxHsW-DuGj4NLA)). The code is implemented in Brian2.  

## Table of Contents
- [Installation](#installation)
- [Quick start](#quickstart)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Installation 
This toolbox is installable via pip with:
```sh
pip install git+https://github.com/imendezguerra/soma_den_mn_model
```

To install it in editable mode, please clone the repository in your project's folder and run:
```sh
pip install -e ./soma_den_mn_model
```
Once the toolbox has been installed you can just import the corresponding packages as:
```python
from soma_den_mn_model.configs import S_MN_Config
```
#### Prerequisites
When installing the package, `pip` will automatically install the required packages stored in `requirements.txt`. 

If you decide to clone the repository, then you can replicate the environment with:
```
conda env create -f environment.yml
```
The file was constructed without the build so it should be compatible with Os, Windows, and Linux.

#### Local setup guide
To set up the project locally do the following:

1. Clone the repository:
    ```sh
    git clone https://github.com/imendezguerra/soma_den_mn_model.git
    ```
2. Navigate to the project directory:
    ```sh
    cd soma_den_mn_model
    ```
3. Create the conda environment from the `environment.yml` file:
    ```sh
    conda env create -f environment.yml
    ```
4. Activate the environment:
    ```sh
    conda activate soma_den_mn_model
    ```

## Quick start 
The package is composed of the following modules:
- `configs.py`: Child dataclasses with properties for S, FR, and FF motor neurons.
- `inputs.py`: Class to generate synaptic excitatory inputs (dynamics of the excitatory neurotransmitter, unitless).
- `pool.py`: Class with functions to simulate a motor neuron pool.

The `tutorials` folder contains examples of how to use the package, including:
- `example_injected_dendrite.ipynb`: Dendrite current injection to a 100 MN pool
- `example_injected_soma.ipynb`: Soma current injection to a 100 MN pool
- `example_synaptic_dendrite.ipynb`: Synaptic excitatory inputs (dendrite compartment) to a 100 MN pool

## Contributing
We welcome contributions! Here’s how you can contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/newfeature`).
3. Commit your changes (`git commit -m 'Add some newfeature'`).
4. Push to the branch (`git push origin feature/newfeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite this repository:

```sh
@software{Mendez_Guerra_soma_den_mn_model,
author = {Mendez Guerra, Irene},
title = {{Soma-dendrite motor neuron model}},
url = {https://github.com/imendezguerra/soma_den_mn_model},
version = {1.0}
}
```
## Contact

For any questions or inquiries, please contact us at:
```sh
Irene Mendez Guerra
irene.mendez17@imperial.ac.uk
```
