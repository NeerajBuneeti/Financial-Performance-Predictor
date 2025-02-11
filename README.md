# Financial Performance Predictor

Financial Analysis

## Overview

This project leverages advanced data analysis and machine learning techniques to predict company financial performance and identify economic trends. Using a comprehensive dataset from Kaggle, we've developed sophisticated models to forecast financial metrics and provide actionable insights for strategic decision-making.

## Features

- 🧹 Data preprocessing and cleaning
- 📊 Exploratory Data Analysis (EDA) with interactive visualizations
- 🔍 Feature engineering and selection
- 🤖 Multiple predictive models:
  - Linear Regression
  - Random Forests
  - Decision Trees
  - Support Vector Machines (SVM)
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
- 🎛️ Hyperparameter tuning for optimized performance
- ✅ Model validation and performance evaluation

## Dataset

The primary dataset used is the "Company Financials Dataset" from Kaggle, which includes:
- Segment information
- Country data
- Product details
- Sales figures
- Profit metrics
- Temporal data (dates and month numbers)

## Key Findings

- Identified crucial indicators of financial performance through regression analysis and EDA
- Developed models with high accuracy in forecasting financial trends
- Uncovered patterns in gross sales across different segments and countries
- Analyzed profit distribution across various markets

Sales Distribution

## Installation

```bash
git clone https://github.com/yourusername/Financial-Performance-Predictor.git
cd Financial-Performance-Predictor
pip install -r requirements.txt
```

## Usage

1. Data Preprocessing:
   ```R
   source("scripts/data_preprocessing.R")
   ```

2. Exploratory Data Analysis:
   ```R
   source("scripts/eda_visualizations.R")
   ```

3. Model Training and Evaluation:
   ```R
   source("scripts/model_training.R")
   ```

4. Hyperparameter Tuning:
   ```R
   source("scripts/hyperparameter_tuning.R")
   ```

## Results

Our models demonstrate strong predictive capabilities:

| Model | Performance Metric | Value |
|-------|---------------------|-------|
| Linear Regression | MSE | 9.36e-07 |
| Random Forest | Variance Explained | 99.63% |
| SVM (tuned) | RMSE | 3016.62 |
| KNN (tuned) | RMSE | 108967409.63 |

Model Comparison

## Future Work

- Integrate real-time data for more responsive predictions
- Explore deep learning models for complex pattern recognition
- Develop a user-friendly interface for non-technical stakeholders

## Project Structure

```
Financial-Performance-Predictor/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── scripts/
│   ├── data_preprocessing.R
│   ├── eda_visualizations.R
│   ├── model_training.R
│   └── hyperparameter_tuning.R
│
├── models/
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   └── svm.pkl
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── images/
│   ├── financial_analysis.png
│   ├── sales_distribution.png
│   └── model_comparison.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

## Contributors

- [Poojith Reddy Annachedu](https://github.com/poojithreddy)
- [Neeraj Vardhan Buneeti](https://github.com/neerajvardhan)
- [Sai Charan Gangili](https://github.com/saicharan)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Illinois Institute of Technology
- Professor Jawahar Panchal
- Kaggle for providing the dataset

---

For more information, please check our [project wiki](https://github.com/yourusername/Financial-Performance-Predictor/wiki) or open an [issue](https://github.com/yourusername/Financial-Performance-Predictor/issues).

[GitHub stars
[GitHub forks

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/33749695/57c68a53-c3dc-4682-beb7-14a101137415/Project_Report.pdf
