# DelRec

This repository contains the code to reproduce the experiments presented in the article [DelRec: learning delays in recurrent spiking neural networks](https://arxiv.org/abs/2509.24852), by Alexandre Queant, Ulysse Rançon, Benoit R Cotterau and Timothée Masquelier.

## Datasets 

Create a `Datasets` folder at the root of the repository.

<details> <summary><strong>For the SSC dataset</strong></summary>
<br>
The Spiking Speech Commands (SSC) dataset contains 35 classes from a larger number of speakers. The number of examples in the train, validation and test splits are 75466, 9981 and 20382 respectively.

The datasets folder should have a subfolder `SSC/`containing `ssc_train.h5`, `ssc_valid.h5` and `ssc_test.h5`, which are all downloadable at: https://zenkelab.org/datasets. The dataloader used comes from https://github.com/dgxdn/ASRC-SNN.

</details>

<details> <summary><strong>For the SHD dataset</strong></summary>
<br>
The Spiking Heidelberg Digits (SHD) dataset containing spoken digits from 0 to 9 in both English and German (20 classes). The train and test sets contain 8332 and 2088 examples respectively (there is no validation set provided).

Just create a subfolder `SHD/`in the `Datasets/` folder, then the Spiking Jelly dataloader we use for this dataset will download and process the data automatically, as seen in https://github.com/Thvnvtos/SNN-delays.

</details>
<br>

Finally, the `Datasets/` folder should have one subfolder `SHD/` and one subfolder `SSC/`.

Then, make sure the configurations in `configs/` have the right `dataset_path`for the corresponding dataset. For example, for the SSC dataset, you need to have:

```text
datasets_path = 'Datasets/SSC'
````

in `configs/perf_SSC.py`.

The PS-MNIST dataset us directly derived from the MNIST dataset, which is already in Pytorch.

## Requirements

This code is based on Pytorch. To install the required libraries, do:
```text
pip install -r requirements.txt
```

## Repository Structure  

```text
configs/           # Parameter & performance configs
notebooks/         # Jupyter notebook for Fig.3
src/               # Core source code
````
On top of that, we provide the scripts for the main experiments:
```text
perf_dataset.py     # Train our model on the given dataset
equiparam_SHD.py    # Training for Fig.3 B and C (top)
penalize_spikes.py  # Training for Fig.3 C (bottom)
```

## Details

<details> <summary><strong>Main Scripts</strong></summary>

| Script | Description |
|--------|-------------|
| `perf_SHD.py` | Train and evaluate the model on the SHD dataset. To reproduce the accuracies displayed in Table 2, run this script with the corresponding SNN model on `seeds = [i for i in range(10)]`.  |
| `perf_PSMNIST.py` | Train and evaluate the model on the PSMNIST dataset. To reproduce the accuracies displayed in Table 1, run this script with the corresponding SNN model on `seeds = [0, 1, 2]`. |
| `perf_SSC.py` | Train and evaluate the model on the SSC dataset. To reproduce the accuracies displayed in Table 1, run this script with the corresponding SNN model on `seeds = [0, 1, 2]`. |
| `equiparam_SHD.py` | Training the different models in order to obtain the evolution of accuracy on the SHD as a function of the number of parameters. To reproduce Fig.3 B and C (top), run this script on `seeds = [0, 1, 2]`. |
| `penalize_spikes.py` | Training the different models while gradually penalizing spikes, in order to have model accuracy as a function of mean firing rate. To reproduce Fig.3 C (bottom), run this script on `seeds = [0, 1, 2]`. The models in this experiment use the same configuration as for `equiparam_SHD`.|

To run one of these scripts, just use :

```text
python perf_SSC.py
````

for example, in the case of the SSC dataset.

We use [Weights and Biases](https://wandb.ai/site) in order to log our metrics and the evolution of parameters. Please add:

```text
os.environ["WANDB_MODE"] = "disabled"
````

at the beggining of the script if you don't want to use wandb. Otherwise, enter your wandb key inn the place-holder:

```text
WANDB_KEY = None # Your key here
```

</details>

<details> <summary><strong>Source Code (<code>src/</code>)</strong></summary> 

| File/Folder | Description |
|-------------|-------------|
| `datasets.py` | Dataset loading scripts for SHD, SSC, PSMNIST. |
| `recurrent_neurons.py` | Implementation of our class of neurons with delays in recurrent connections. |
| `utils.py` | Helper functions (metrics, seeding, logging ...) |
| `SSC/` `PSMNIST/` `SHD/`| Dataset-specific processing scripts for each dataset. In each subfolder, you have `snn` provides the implementation of the different networks that can be used on the dataset, and `trainer` implements the function used for the training of the networks. `SSC/`and `PSMNIST/`use the same `snn` script.|

</details> 

<details> <summary><strong>Training time</strong></summary> 

On a NVIDIA A100 GPU, the training time on one seed for the number of epochs detailed in the corresponding config is of the order of:

- SSC ~ 1 day for 100 epochs.
- PSMNIST ~ 2 days for 200 epochs.
- SHD ~ 1 hour for 150 epochs.

</details>

## License 

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
