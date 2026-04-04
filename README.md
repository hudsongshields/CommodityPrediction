# DS-TGNN: Spatiotemporal Commodity Prediction Framework

## Project Overview
The Deep Spatiotemporal Graph Neural Network (DS-TGNN) is a research framework designed to predict commodity price returns by analyzing global meteorological data. The system utilizes a **Score-Based Generative Model** to identify structural patterns in noisy weather data, providing a robust foundation for multi-asset forecasting.

## Core Technical Concepts

### 1. Score-Matching and the ScoreNetwork
In this framework, a "Score" refers to the gradient of the log-density of the data. Essentially, the model learns the "shape" of typical meteorological patterns. 
- **Score Matching**: A training objective where the model learns to identify where a noisy data point "should" move to return to the stable data manifold (the range of realistic values).
- **Fisher Divergence**: The mathematical metric used to measure the difference between the model's predicted gradients and the true structure of the data. Minimizing this divergence ensures the model's logic aligns with physical reality.

### 2. Denoising via Tweedie’s Formula
The framework uses **Tweedie's Formula**, a statistical tool that allows the model to "clean" a noisy observation. By calculating the most likely true value of a weather variable (e.g., precipitation) given a noisy reading, the model can make predictions based on stable signals rather than random fluctuations.

### 3. Spatiotemporal Graph Neural Networks (GNN)
Commodity markets are interconnected. The price of Corn affects Cattle feed, which in turn influences Beef prices. 
- **GNN**: A neural network architecture that understands these relationships. It treats each commodity and meteorological hub as a "node" in a graph, allowing information about a supply shock in one region to flow logically to related assets.

### 4. Variance Exploding SDE (VE-SDE)
The model follows a stochastic framework that systematically adds noise to data during training to learn how to remove it. This "Variance Exploding" approach is a standard in generative modeling that helps the network become resilient to extreme weather volatility.

## Research Methodology

### Evaluation and the DBA Benchmark
To ensure the model provides real-world value, performance is measured against the **DBA (Invesco DB Agriculture Fund)**. This is a standard institutional ETF that tracks a diversified basket of agricultural commodities. 
- **Excess Return**: The performance of the model minus the performance of the DBA benchmark.
- **Information Ratio (IR)**: A metric measuring the consistency of returns relative to risk. An IR of 0.81 indicates strong risk-adjusted outperformance.

### Embargoed Data Splitting
To prevent "look-ahead bias" (where a model accidentally sees future data during training), we use **Embargoed Splitting**. This ensures a mandatory 30-day gap between the training data and the testing data, accounting for the overlapping nature of the 30-day return windows.

## Implementation Structure

- **models/ds_tgnn.py**: The primary architecture combining the GNN and the Score-Matching features.
- **models/diffusion/**: The internal logic for the ScoreNetwork and Fisher Divergence loss functions.
- **scripts/fetch_global_weather.py**: Data ingestion tool for historical meteorological records.
- **scripts/generate_ultimate_research_suite.py**: The visualization engine for the four authoritative performance dashboards.
- **evaluate_experiments.py**: The main research entry point for training and backtesting.

## Quick Start
To reproduce the latest research results and generate the performance dashboards:
1. Ensure dependencies are installed (PyTorch, PyTorch Geometric, YFinance).
2. Execute the evaluation script:
   ```bash
   python evaluate_experiments.py
   ```
3. View results in the `plots/` directory.
