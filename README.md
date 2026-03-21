# Collision Severity Analysis In Canadian Road Accidents

## Project Overview

This project investigates the determinants of collision severity and injury outcomes in Canadian road accidents using national collision data from 1999 to 2014.

The objective is to analyze how environmental conditions, roadway characteristics, vehicle attributes, safety device usage, and road user type influence the likelihood of fatal collisions and severe injury outcomes.

Data Source: Transport Canada – National Collision Database (NCDB)

---

## Research Questions

### RQ1: Environment and Traffic Context Factors Affecting Crash Severity

How do road, weather, and traffic control conditions influence the likelihood of fatal collisions in Canada between 1999 and 2014?

Hypotheses:

- H₀₁: Collision severity is independent of road, weather, and roadway configuration.
- H₁₁: Adverse road and weather conditions significantly increase the likelihood of fatal collisions.
- H₀₂: Traffic control and roadway configuration have no effect on fatality risk.
- H₁₂: Specific roadway configurations and traffic control conditions significantly affect fatal collision likelihood.

---

### RQ2: Human, Vehicle, and Safety Factors

To what extent do vehicle characteristics, safety device usage, and road user type affect injury severity outcomes?

Hypotheses:

- H₀₃: Safety device usage does not affect injury severity.
- H₁₃: Safety device usage significantly reduces the likelihood of severe or fatal injuries.
- H₀₄: Injury severity does not differ by road user type or vehicle type.
- H₁₄: Vulnerable road users and certain vehicle types experience higher injury severity.

---

## Dataset Description

The National Collision Database contains collision-level, vehicle-level, and person-level records including:

Collision-level variables:
- Weather conditions (C_WTHR)
- Road surface condition (C_RSUR)
- Road alignment and configuration (C_RALN, C_RCFG)
- Traffic control (C_TRAF)
- Collision severity (C_SEV)

Person-level variables:
- Injury severity (P_ISEV)
- Safety device usage (P_SAFE)
- Road user type (P_USER)
- Age and sex

Vehicle-level variables:
- Vehicle type (V_TYPE)
- Vehicle model year (V_YEAR)

The dataset is primarily categorical and required preprocessing including imputation and encoding.

---

## Methodology

### Data Preparation

1. Selected relevant variables for RQ1 and RQ2.
2. Cleaned missing and unknown values.
3. Encoded categorical variables using one-hot encoding.
4. Created binary target variables:
   - y_fatal (fatal vs non-fatal collision)
   - y_severe_injury (severe vs non-severe injury)
5. Split data into training and testing sets using stratified sampling.

---

### Modeling Approaches

Classical Machine Learning Models:

- Logistic Regression (primary model for interpretation)
- Decision Tree Classifier (secondary model for interaction insights)

Handling Class Imbalance:
- Class weighting applied during model training.
- Evaluation focused on recall and ROC-AUC rather than accuracy alone.

Deep Learning Extension:

- A feedforward neural network (MLP) was implemented to evaluate potential non-linear relationships.
- The same preprocessing pipeline was reused.
- Model trained using binary cross-entropy and class weights.
- Deep learning results were compared to classical models.

---

## Results Summary

RQ1 Findings:

- Adverse weather and certain roadway configurations are associated with increased fatal collision risk.
- Traffic control conditions show measurable influence on severity outcomes.
- Logistic regression provided interpretable coefficients aligned with policy relevance.

RQ2 Findings:

- Vulnerable road users (e.g., pedestrians, cyclists, motorcyclists) exhibit higher severe injury rates.
- Safety device usage is associated with reduced severe injury probability.
- Vehicle characteristics contribute to injury severity variation.

Model Evaluation:

Performance metrics included:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrices

Logistic regression was retained as the primary model due to interpretability and policy applicability.

---

## Repository Structure

