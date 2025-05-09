# Refining Decision Boundaries In Anomaly Detection Using Similarity Search Within the Feature Space

Anomaly detection in highly imbalanced datasets, where anomalies are rare and diverse, presents significant challenges in machine
learning. Active learning, a strategy that optimally queries for labels, can significantly enhance anomaly detection by reducing
labeling costs and improving model performance. Traditional active learning methods rely primarily on querying an oracle to label
uncertain points, often overlooking the intrinsic structure in the feature space of the data for model refinement. This paper presents
a novel anomaly detection framework based on a Sparse Adversarial Attention-Based AutoEncoder (SDA²E), focusing on critical
features in imbalanced data through sparsity and attention mechanisms. The model is enhanced with active learning, integrating
three innovative strategies that utilize similarity search within the feature space to improve detection accuracy and robustness.
The first strategy identifies normal-like points using similarity search with oracle-labeled normals, incorporating these points into
the training set to refine the AutoEncoder’s reconstruction capabilities. The second strategy prioritizes points similar to oracle-
labeled anomalies in the ranked anomaly detection list, improving detection precision. The third, hybrid strategy combines both
approaches, balancing anomaly ranking with robust model retraining. Through theoretical formulations, algorithmic design, and
experimental validations, we demonstrate the framework’s effectiveness in detecting anomalies efficiently with a small oracle query
budget. Our results show how this approach reduces reliance on extensive labeled datasets and overcomes the challenges inherent
in imbalanced data. This work highlights the potential of integrating sparse AutoEncoders with similarity-based active learning as
a robust solution for anomaly detection tasks, with a particular emphasis on tracking advanced cyber threats like APTs.

# Architecture 
![Alt text](https://github.com/aladaen/aladaen-apt/blob/main/figures/aladaenarchitecture.png "General Architecture")


# Workflow 

![Alt text](https://github.com/aladaen/aladaen-apt/blob/main/figures/aladaenworkflow.png "General Architecture")

# ADAEN APT-AutoEncoders 

Run the script:
```shell
root % cd src
src % python main.py [model type: AE, AAE, ADAE] [data file] [labels file]
EXAMPLE:
src % python main.py AE pandex/trace/ProcessAll.csv pandex/trace/trace_pandex_merged.csv
```

# Baseline anomaly detectors

## Quick start

# Dependencies

```
pip install pandas numpy scipy matplotlib scikit-learn pillow h5py pyfpgrowth tensorflow keras rpy2
```


Also, to use the R-based anomaly detection method for rare rule mining, R >= 3.3.3 must be
installed and the following libraries should be installed by running
the following command within R:
```
install.packages(c("stringr","ppls","plyr"))
```


# Running

## Input data

The different anomaly detectors presented below use as input the process centric formal context represented in CSV formats. These data files are publically available in
```
https://gitlab.com/adaptdata
```
## Algorithms

### AVF (Attribute Value Frequency) anomaly detection

As described in the original paper authored by Koufakou et al. 2007, the AVF scores are averaged probabilities of attribute values.
The script `ad.py` implements this anomaly detection method in a batch and a streaming mode.  The batch mode reads and
processes all of the input and then traverses it again to score it, whereas the streaming mode traverses the input just once
and scores each transaction when it is first seen, using the model obtained for the previous input consumed only.
 
The script's options are as follows.

```
usage: ad.py [-h] -i INPUT -o OUTPUT -s avf
             [-m {batch,stream}]
```

Batch scoring is the default mode with this algorithm.
 
### Frequent Pattern Outlier Factor (FPOF) and Outlier Degree (OD)

We have also incorporated implementations of two simple anomaly detectors from the
literature, called FPOF (Frequent Pattern Outlier Factor) and OD (Outlier
Degree):

```
pattern.py [-h] --input INPUT --output OUTPUT [--score {fpof,od}]
                  [--conf CONF] [--minsupp MINSUPP]
```

Here, the `--score` parameter chooses which algorithm to use, `conf` chooses the
confidence level to use for rules and `--minsuppo` chooses the minimum support
level to use.  Both of these take values between 0 and 1; higher tends to be
faster.  `conf` is only relevant for `--score od` while `--minsupp` is
relevant to both algorithms.  

### OC3/Krimp:  

We have also added a Linux binary based on the Krimp system, which mines the
frequent (closed) itemsets and then attempts to identify a subset of them that
"compress the data well".  The end result includes scores for the objects
indicating their "compressed size", which we interpret
as anomaly scores.  There are currently no parameters (Krimp does have some,
but we currently hard-wire them to reasonable values).

```
krimp.py [-h] --input INPUT --output OUTPUT
```

### CompreX:  

We used the Matlab implementation of CompreX available at the official web page of the author:

```
http://www.andrew.cmu.edu/user/lakoglu/tools/CompreX_12_tbox.tar.gz
```
Note that we have adapted the original Comprex code for accepting our input formal context files, and we would provide the modified version of th code we used upon request.

## Checking scores against ground truth

The script `check.py` takes a score file (produced for example by `ad.py` or `krimp.py`),
a ground truth CSV file, and some additional options and produces a
ranking file, which is a CSV file listing just those object IDs found
in the ground truth with their ranks (= how anomalous the object
was).  The checker also prints out the "normalized discounted
cumulative gain" (NDCG) and the "area under ROC curve" (AUC) as
proxies for how "good" the ranking is overall.
The options are as follows:

```
check.py [-h] -i INPUT -o OUTPUT -g GROUNDTRUTH
                [-t TY] [-r]

```

The optional argument `-t TY` allows to specify the
"type" of ground truth ID.  The ground truth CSV files consist of
objectID,type pairs, where the type is a string indicating what kind
of obejct the objectID is.  For example, in the ADM ground truth CSV
files the types are things like `AdmSubject::Node`.  It is helpful to
specify this so that the checker can filter out the ground truth
objects that we do not expect to find in a given context.  If such
objects are included then the NDCG and AUC scores are typically very
low due to the other objects being counted as "misses".

The optional `-r` argument allows to specify whether the scores should
be sorted in decreasing order (i.e. high score = more anomalous).  The
default behavior is to sort in increasing order, which is appropriate
for AVF (small score = more anomalous).


### APT-GAN (Adversarial Generative Neural Networks) anomaly detection

Here we are going to use a GAN to train a neural network for generating fake APT data, then use the generated data to test a given anomaly detection method.
The script APT-GAN/skynet.py trains a GAN using a formal context, thun runs a rule-mining anomaly detection method.
The script's options are as follows.

```GAN.py  -i ./sample/ProcessEvent.cs -g ./sample/cadets_pandex_merged.csv -t 1000```

The results are generated in APT-GAN/output folder, and show the variation of the ndcg scores with the simulated data.

## Running example

Let's run the AVF anomaly detector on the ProcessEvent context (BSD, 1st attack scenario) available in the sample folder (the original file can be downloaded from the data repository at https://gitlab.com/adaptdata/e2/blob/master/pandex/cadets/ProcessEvent.csv.gz).

The user can then call the python script:

```python3 ad.py -i ./sample/ProcessEvent.csv -o ./sample/ScoreProcessEvent.csv -s avf -m batch```

As explained above, this script produces a scoring file where an anomaly score is calculated for each process in the formal context.
Then using the check.py script and the ground truth database (./sample/cadets_pandex_merged.csv), the previously generated scoring file is ranked and the global nDCG as-well as the AUX are calculated for this context.

```python3 check.py -i  ./sample/ScoreProcessEvent.csv -o OutputProcessEvent.csv -g ./sample/cadets_pandex_merged.csv -t AdmSubject::Node```

# Some results 
## AutoEncoders Reconstruction Error

![Alt text](https://github.com/flagus-apt/flagus/blob/main/figures/Figure_ReconstructionError.png "AutoEncoders Reconstruction Error")

## Active Learning BSD Pandex

![Alt text](https://github.com/flagus-apt/flagus/blob/main/figures/AL_BSD_Pandex.png "Active Learning BSD Pandex")

## Active Learning BSD Bovia

![Alt text](https://github.com/flagus-apt/flagus/blob/main/figures/AL_BSD_Bovia.png "Active Learning BSD Bovia")

## Active Learning Windows Pandex

![Alt text](https://github.com/flagus-apt/flagus/blob/main/figures/AL_Windows_Pandex.png "Active Learning Windows Pandex")

## Active Learning Windows Bovia

![Alt text](https://github.com/flagus-apt/flagus/blob/main/figures/AL_Windows_Bovia.png "Active Learning Windows Bovia")

## Active Learning Linux Pandex

![Alt text](https://github.com/flagus-apt/flagus/blob/main/figures/AL_Linux_Pandex.png "Active Learning Linux Pandex")

## Active Learning Linux Bovia

![Alt text](https://github.com/flagus-apt/flagus/blob/main/figures/AL_Linux_Bovia.png "Active Learning Linux Bovia")

## Active Learning Android Pandex

![Alt text](https://github.com/flagus-apt/flagus/blob/main/figures/AL_Android_Pandex.png "Active Learning Android Pandex")

## Active Learning Android Bovia

![Alt text](https://github.com/flagus-apt/flagus/blob/main/figures/AL_Android_Bovia.png "Active Learning Android Bovia")


