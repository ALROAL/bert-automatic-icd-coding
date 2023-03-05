# BERT-based Automatic ICD Coding
Automatic ICD coding is the task of assigning codes from the International Classification of Diseases (ICD) to medical notes. These codes describe the state of the patient and have multiple applications, e.g., computer-assisted diagnosis or epidemiological studies. ICD coding is a challenging task due to the complexity and length of medical notes.

- [Quick start](#quick-start)
- [Description](#description)
- [Usage](#results)
- [Results](#report)
- [Weights & Biases](#weights--biases)

## Quick start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download data**

Sign the data use agreement and download MIMIC-III dataset from [PhysioNet](https://mimic.physionet.org).

Organize your data using the following structure

```
data
└───raw/
|   |   NOTEEVENTS.csv
|   |   DIAGNOSES_ICD.csv
|   |   PROCEDURES_ICD.csv
|   |   D_ICD_DIAGNOSES.csv
|   |   D_ICD_PROCEDURES.csv 
|   |   ICD9_descriptions
└───split/
|   |   *_hadm_ids.csv
```

`ICD9_descriptions` is available [here](https://github.com/jamesmullenbach/caml-mimic/blob/master/mimicdata/ICD9_descriptions), and 
`*_hadm_ids.csv` are available [here](https://github.com/jamesmullenbach/caml-mimic/tree/master/mimicdata/mimic3).

**3. Process and prepare data**

Run the ```prepare_data.py``` script.
```bash
python prepare_data.py
```

## Usage

Specify the experiment configuration in ```config.yaml``` and run ```train.py``` script.
```bash
python train.py
```

## Description and Results

![network architecture](https://i.imgur.com/VTAkpjh.png)

## Weights & Biases
The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/). Training and validation loss curves are logged to the platform.
When launching a training, a link will be printed in the console. Click on it to go to your dashboard. If you have an existing W&B account, you can link it by setting the `WANDB_API_KEY` environment variable. If not, it will create an anonymous run which is automatically deleted after 7 days.

## Acknowledgements
The code is based on the following two great repositories
- https://agit.ai/jsx/MCA_BERT
- https://github.com/MiuLab/PLM-ICD
