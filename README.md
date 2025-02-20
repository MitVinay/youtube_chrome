# youtube-chrome

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Process : https://docs.google.com/document/d/1E1JoHTwV-CqYrivomhTK-zohAmt5GLQemSRO6I9WIlA/edit?tab=t.0

Plugin 

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

# Goal
The goal of the project is to develop a Youtube Chrome Plugin, This extention will empower the youtube influencer to understand the audience anc create content to increase the rech withoout paying a penny to makreting agencies. 

# Vision 
**You Not Just get TO the Audience but Get THROUGH the Audience**

YouTube influencers face a significant challenge: understanding audience preferences and content consumption patterns to boost engagement. While marketing agencies and advanced analytics tools provide data-driven insights, these solutions are often prohibitively expensive for many influencers.
As a result, influencers are left to manually analyze vast volumes of audience comments, a process that is not only time-consuming but also error-prone. With thousands of comments pouring in on each video, it becomes nearly impossible to identify and act on critical insights buried within the feedback. These missed opportunities could have otherwise informed content strategies, strengthened audience connections, and accelerated channel growth.
This tedious and inefficient process consumes valuable time and often leads to overlooked feedback that could have been instrumental in understanding audience needs and preferences. Influencers need a smarter, more accessible solution to unlock these insights efficiently and effectively.
The YouTube plugin application addresses two critical business metrics: marketing cost reduction and time efficiency.


# Appraoch 

**Initial Phase:**

Data Collection → Exploratory Data Analysis → Baseline Model → Experiment Tracking

**Second Phase:**

Monolithic Architecture:

Data Ingestion → Data Preprocessing → Model Training → Model Evaluation → Model Registration

**Third Phase:**

Develop a Flask API for model serving, and a plugin.


# Roadmap

First, I will provide a quick overview of the tool stack I will be using, followed by where each tool will be applied. I will also include the key libraries at a high level.

## 1. Tool Stack Used:

Data Versioning: DVC
Data Storage: DVC
Code Versioning: GitHub
Experiment Tracking: MLflow
Model Registry: MLflow
Hosting MLflow: Dagshub
CI/CD Workflow: GitHub Actions
Containerization: Docker
API: Flask
IDE: Visual Code

As discuss lets discuss more about phase one and phase 2.

## 2. Phase 1 Execution  

We will use **Google Colab, MLflow, and Dagshub** to execute this phase.  

## Phase 1 Flow  
**Data Collection → Exploratory Data Analysis → Baseline Model → Experiment Tracking**  

## Data Collection  
The training and testing data is collected from **Kaggle**.  

- **Dataset Link:** [Provide Link]  
- **Summary:**  
  - The dataset contains **comments and sentiment labels**.  
  - There are **three sentiment categories**: **Neutral, Positive, and Negative**.  

## Exploratory Data Analysis  
The primary libraries used for data cleaning and visualization are:  
- **textacy**  
- **pyLDAvis**  
- **WordCloud**  

## Data Preprocessing Steps  
- **Duplicate and Null Value Handling**  
- **Text Normalization**  
- **Whitespace Removal**  
- **HTML Tags Removal**  
- **Punctuation Removal**  
- **Bracket Removal**  
- **Emoji Removal**  
- **Length-Based Filtering**  
- **Non-Meaningful Row Filtering**  

## Summary of Insights  
- The dataset is **imbalanced**, so we need to apply techniques like **ADASYN** to address this issue.  
- **Stop words removal** was performed, but words like *not, but, or* (negative conjunctions) were **kept** to preserve sentiment.  
- Several **non-English words** were found in the dataset.  
- The **label distribution** does not vary significantly with **word count**. However, after **6,000 words**, most comments are **positive**, indicating that **longer comments tend to be positive**.  
- **Word cloud analysis** revealed a high presence of **political comments**, suggesting that the trained model will likely perform better on **political YouTube channels**.  
