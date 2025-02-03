# PII-Data-Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blueviolet)
![Medal](https://img.shields.io/badge/Medal-Bronze-bronze)

## Project Overview

This project tackles the Kaggle challenge of detecting Personally Identifiable Information (PII) in student essays using advanced NLP techniques.

### Challenge Details
- **Objective**: Identify 7 types of PII in student essays
- **PII Types**: 
  - Student Name
  - Address
  - Phone Number
  - Email
  - Website
  - ID Number
  - Username

### Classification Scheme
- 14 total classes using BIO (Beginning, Inside, Outside) tagging
- Tokens labeled as:
  - `B-{TYPE}`: First word of multi-word PII
  - `I-{TYPE}`: Subsequent words of multi-word PII
  - `O`: Non-PII tokens

### Key Challenges
- Extreme class imbalance
- Minimal PII tokens in most essays
- Evaluation metric: F5 score (Recall weighted 5x more than Precision)

## Solution Approach

### Methodology
- Ensemble of DeBERTa models
- Techniques to address class imbalance:
  - Adaptive thresholding
  - Negative undersampling
  - Model ensembling

### Performance
- F5 Score: 0.96
- Kaggle Ranking: 173rd out of 2,100 teams
- **Achievement**: Bronze Medal 🥉

## Acknowledgments
- Kaggle Competition Organizers
- DeBERTa Model Creators