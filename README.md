
## Source Code for EI-kMean Space Partition for Drift Detection
Hello! This is the python code for EI-kMeans Drift deteciton presented in the paper "Concept Drift Detection via Equal Intensity k-Means Space Partitioning" ([https://arxiv.org/abs/2004.11587](https://arxiv.org/abs/2004.11587))

## Getting Started

### Project Structure
```
.
+-- README.md
+-- kMeansChi2_lib.py
+-- kMeansChi2_Exp3.py
+-- kMeansChi2_Exp2.py
+-- EIkMeans_lib.py
+-- EIkMeans_Exp3.py
+-- EIkMeans_Exp2.py
+-- Datasets
|   +-- Exp1_Demo
|   |   +-- ParititionDemo1_Gaussian.csv
|   |   +-- ParititionDemo2_ThreeGaussian.csv
|   +-- OriginalDataFiles
|   |   +-- 1_Higgs
|   |   |   +-- data0.mat
|   |   |   +-- data1.mat
|   |   +-- 2_MiniBooNe
|   |   |   +-- MiniBooNE_PID.txt
|   |   +-- 3_Arabic_Digit
|   |   |   +-- ArabicDigit_Shuffled_With_Sex.csv
|   |   +-- 4_Localization
|   |   |   +-- ConfLongDemo_JSI.txt
|   |   +-- 5_Insects
|   |   |   +-- Insects.data
|   +-- Exp2_Syn_dataGenerator.py (**Step 1**)
|   +-- Exp3_Rea_dataGenerator.py (**Step 2**)
```

### Prerequisites
```
sklearn
scipy
numpy
matplotlib
```

## Running the tests

To run the test on the given datasets you need to generate data samples from their original datasets.
Please go to ```Datasets``` folder and run

```
# Step 1. run this code to generate synthetic data for exp 2
python Exp2_Syn_dataGenerator.py

# Step 2. run this code to generate synthetic data for exp 3
python Exp3_Rea_dataGenerator.py
```
Now you should be able to reproduce the experiment by running the code located under the main folder.
```
python kMeansChi2_Exp3.py
python kMeansChi2_Exp2.py
python EIkMeans_Exp3.py
python EIkMeans_Exp2.py
```

## Detect Concept Drift

Given data batches ```Batch_train``` and ```Batch_test``` and the desired signficance level ```alpha```
```python
import EIkMeans_lib as eikm

num_train = Batch_train.shape[0]
num_test = Batch_test.shape[0]

# According to the constrains of Pearson's Chi-square Test
# It's better to keep no less than 50 samples in each partition
num_partitions = num_train / 50

# Initialize EI-kMeans instances and build the histogram (partitions)
eikm_instance = eikm.EIkMeans(num_partitions)
eikm_instance.build_partition(Batch_train, num_test)

# Performing concept drift detection
eikm_instance.drift_detection(Batch_test, alpha)
```
### Remark

You may need to perform normalization on your datasets to have a stable drift detection performance.

## Authors

* **Anjin Liu** Postdoctoral Research Associate, Anjin.Liu@uts.edu.au
* **Jie Lu** Distinguished Professor, Jie.Lu@uts.edu.au
* **Guangquan Zhang** Associate Professor, Guangquan.Zhang@uts.edu.au

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

The work presented in this paper was supported by the Australian Research Council (ARC) under Discovery Project DP190101733.