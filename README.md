# Integrating Alz-SDoH info from EHR and literature via knowledge graphs and LLMs 
This repository holds the official code for the paper below:
> A novel computational analysis integrating social determinants information from EHR and literature with Alzheimer’s disease biological knowledge through large language model and knowledge graphs

## 🦸‍ Abstract
**Background and Objectives:** By 2050, Alzheimer’s disease and related dementias (ADRD) are projected to affect over 100 million individuals worldwide, posing significant challenges for patients and substantial burdens on public health systems. Social determinants of health (SDoH) are nonmedical factors influencing individuals’ health outcomes including the risk of AD. The determinants encompass the conditions in which a person is born, grows, works, lives, as well as the person’s age and the broader set of factors shaping daily life. Although growing literature evidence suggests the impact of SDoH on AD is crucial for disease mechanism investigation and interventions, the current understanding of it remains very limited.
**Research Design and Methods:** In this study, we present a systematic analysis to leverage the recent advancements of large language models (LLMs) and knowledge graphs to extract AD-related SDoH knowledge from scientific literature and electronic health records (EHR), integrating this knowledge further into biological knowledge for AD etiology research through knowledge graph (KG) construction and graph deep learning. We perform KG-link predictions for knowledge discovery and validate our results using multimodal biological data from single-cell RNA-seq and proteomics experiments. 
**Results:** We generated an SDoH knowledge graph with around 92k triplets, integrating literature and EHR data. Through various link prediction experiments, we observed higher accuracy when integrating SDoH into knowledge graphs. Additionally, exploratory predictions uncovered potential SDoH-gene interactions, many of which were validated through differential expression analysis using proteomics and RNA-seq data. 
Discussion and Implications: We conducted a novel knowledge graph-based analysis that integrates SDoH from both literature and EHR data to enhance link prediction in AD-related biomedical networks. Our findings highlight the potential interplay between social determinants and biological predispositions in AD and may guide more personalized, socially aware healthcare interventions.

## 📝 Install
The framework is implemented in Python. Please make sure to install the related packages, use
```bash
pip install -r requirements.txt
```

## 🔨 Usage
The main program entrance is the :
```cmd

```

## :file_cabinet: Data
The EHR data we used in this study are collected from the MIMIC-III Clinical Database <https://physionet.org/content/mimiciii/1.4/>; and the literature data are obtained from PubMed search engine. The proteomics data we used are collected from the Accelerating Medicines Partnership for Alzheimer’s Disease (AMP-AD) Diverse Cohorts and the Religious Orders Study and Memory and Aging Project (ROSMAP), both available in Synapse (AMP-AD: https://www.synapse.org/Synapse:syn59611693; ROSMAP: https://www.synapse.org/Synapse:syn21261728). The single-cell RNA sequencing data are publicly available from the Seattle Alzheimer's Disease Brain Cell Atlas (SEA-AD) and downloaded from https://cellxgene.cziscience.com/collections/1ca90a2d-2943-483d-b678-b809bf464c30. 

        
### 🤝 Acknowledgements
This work was supported in part by the NIH grants P30 AG073105, U01 AG066833, U01 AG068057, U19 AG074879, and R01 AG071470.

