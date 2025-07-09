# γ<sub>SRME</sub>: Grüneisen parameter benchmark test for foundational machine-learning potentials

γ<sub>SRME</sub> employs foundation Machine Learning Interatomic Potentials and phonopy to determine the Grüneisen parameter responsible for the thermal expansion and compare them to DFT reference data.

# Install 
Clone repository:
```
git clone https://github.com/MPA2suite/gamma_SRME.git
```
Then install in editable mode:
```
pip install -e .
```

# Usage
The example scripts showcase a sample workflow for testing a MACE potential and comparing the Grüneisen parameter with DFT calculations for a collection of different materials. The scripts may be modified easily to use any foundation Machine Learning Interatomic Potentials. 

Example scripts are found in the `scripts` folder. Model results and scripts are found in the `models` folder. 

To obtain Grüneisen parameter results you may run the workflow on a GPU job, as phonopy has relatively low CPU cost. The `1_test_srme.py` script calculates the displaced force sets and the Grüneisen parameter for each material. The script also supports job arrays outputting one file per array task, which are collected in the evaluation script.

The `2_evaluate.py` script evaluates the predictions, collecting the array task files and printing the results both to the terminal and to a file. The `gamma_srme.json.gz` output file contain additional information about the model run, which can be read as a pandas DataFrame for further analysis.




# How to cite

```
@misc{póta2024thermalconductivitypredictionsfoundation,
      title={Thermal Conductivity Predictions with Foundation Atomistic Models}, 
      author={Balázs Póta and Paramvir Ahlawat and Gábor Csányi and Michele Simoncelli},
      year={2024},
      eprint={2408.00755},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2408.00755}, 
}
```
