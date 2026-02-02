# Blok Interview Prep: Persona-Based Behavioral Simulation

## Overview

This project demonstrates a foundational pipeline for behavioral simulation using customer personas derived from e-commerce transaction data. Built as interview preparation for [Blok](https://www.joinblok.co/), a behavioral simulation engine company, it bridges the gap between static customer segmentation and dynamic agent-based modeling.

**Core thesis**: Behavioral clustering provides the structural foundation for agent instantiation. By extracting latent behavioral patterns from real-world data, we can construct persona profiles that serve as initialization parameters for LLM-powered agents capable of simulating realistic customer decision-making.

## Motivation

Traditional customer segmentation yields descriptive insights (e.g., "high-value, infrequent buyers"), but doesn't capture the decision heuristics, contextual preferences, or cognitive patterns needed for predictive simulation. This project explores how to:

1. **Map behavioral data to actionable personas** — Identify clusters that represent meaningfully distinct behavioral archetypes, not just demographic groupings
2. **Translate static profiles into dynamic agents** — Use persona attributes (purchasing cadence, price sensitivity, category preferences) as behavioral priors for Claude-powered agents
3. **Validate simulation fidelity** — Test whether persona agents respond to product scenarios in ways consistent with the cluster behaviors they represent

This approach mirrors Blok's methodology: grounding synthetic user agents in real behavioral patterns to enable predictive, rather than purely retrospective, product testing.

## Dataset

**[Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)** (Kaggle)

100k orders from 2016–2018 with rich behavioral signals: purchase frequency, basket composition, payment preferences, review sentiment, and delivery experience. The granularity enables clustering by behavioral tendencies rather than demographics.

### Setup Instructions

1. **Install Kaggle CLI** (in your virtual environment):
   ```bash
   pip3 install kaggle
   ```

2. **Configure Kaggle API credentials**:
   - Go to https://www.kaggle.com/settings/account
   - Scroll to "API" section and click "Create New Token"
   - This downloads `kaggle.json` with your credentials
   - Move it to the correct location:
     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

3. **Download and extract the dataset**:
   ```bash
   kaggle datasets download -d olistbr/brazilian-ecommerce -p ./data/raw
   unzip -o ./data/raw/brazilian-ecommerce.zip -d ./data/raw
   ```

## Workflow

### Phase 1: Exploratory Data Analysis
[`01_eda_behavioral_clustering.ipynb`](notebooks/01_eda_behavioral_clustering.ipynb)

Explore the raw transaction data to understand distributions, identify behavioral signals, and assess data quality. Examines purchase patterns, payment methods, review behavior, and delivery outcomes.

### Phase 2: Feature Engineering
[`02_feature_engineering_clustering.ipynb`](notebooks/02_feature_engineering_clustering.ipynb)

Transform raw transactions into customer-level behavioral features: purchase frequency, monetary value, basket size, installment usage, credit card preference, category diversity, review sentiment, and shopping timing.

### Phase 3: Clustering
[`03_clustering.ipynb`](notebooks/03_clustering.ipynb)

Apply K-means clustering to identify distinct behavioral segments. Evaluate cluster quality using elbow method and silhouette scores. Final model: 7 clusters across 93,357 customers.

### Phase 4: Persona Profiling
[`04_persona_profiling.ipynb`](notebooks/04_persona_profiling.ipynb)

Characterize each cluster with representative statistics and z-scores. Generate natural language persona descriptions and decision heuristics. Output: structured persona profiles with LLM-ready system prompts.

### Phase 5: Agent Simulation
[`05_agent_simulation.ipynb`](notebooks/05_agent_simulation.ipynb)

Instantiate Claude-powered agents from persona profiles. Run 6 product scenarios across all 7 personas (42 simulations). Validate that agent responses align with underlying behavioral profiles.

## Results

- **7 distinct personas** derived from real behavioral patterns (e.g., "High-Value Financing Shopper", "Critical Shopper", "Cash Customer")
- **100% validation alignment** — personas respond consistently with their cluster characteristics
- **Clear differentiation** — Critical Shopper rejected 6/6 scenarios; High-Value Financing Shopper accepted 5/6

## Scope

**Included**: EDA, feature engineering, clustering pipeline, persona profiling, Claude API agent instantiation, scenario simulation, validation framework

**Out of scope**: Production deployment, real-time backtesting, multi-agent interactions, calibration against historical conversion rates

## Why This Matters for Behavioral Simulation

Static personas are descriptive; agent-based personas are *generative*. By grounding LLM agents in empirically-derived behavioral clusters, we can predict responses to novel scenarios, explore counterfactuals before committing engineering resources, and identify edge-case behaviors that might respond unpredictably to standard interventions.

Customer clustering becomes the foundation for a simulation engine that can compress weeks of A/B testing into hours of agent-based experimentation.
