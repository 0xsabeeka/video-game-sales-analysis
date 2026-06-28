# Video Game Sales Analysis

A Streamlit web application for statistical analysis of video game sales using Python.

This project explores the question: **Do violent video games sell more?**
It analyzes video game sales using ESRB ratings, global sales data, descriptive statistics, probability distributions, hypothesis testing, and regression modeling.

## Project Overview

The project uses a video game sales dataset with ratings and sales information. The analysis focuses on comparing sales performance across ESRB rating categories such as E, E10+, T, and M.

The main goal is to understand whether M-rated games show higher sales patterns and whether rating alone is a strong factor in predicting sales.

## Features

* Interactive Streamlit web app
* Dataset overview and raw data preview
* Graphical analysis of ratings, genres, platforms, and sales
* Descriptive statistics by ESRB rating
* Confidence interval visualization
* Probability distribution fitting
* Hypothesis testing using Welch's t-test
* Regression and prediction analysis
* Final conclusion based on statistical results

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* Scikit-learn

## Dataset

The project uses the **Video Game Sales with Ratings** dataset.

Main variables include:

* Game name
* Platform
* Genre
* Global sales
* Regional sales
* Critic score
* User score
* ESRB rating

## Project Structure

```text
video-game-sales-analysis/
│
├── app.py
├── logo.png
├── requirements.txt
├── Video_Games_Sales_as_at_22_Dec_2016.csv
└── .streamlit/
    └── config.toml
```

## How to Run

1. Install the required Python libraries:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:

```bash
streamlit run app.py
```

3. Open the local Streamlit URL in your browser.

## Main Analysis

The project includes:

* Sales comparison across ESRB ratings
* Rating distribution analysis
* Genre and platform sales analysis
* Correlation heatmap
* Box plot analysis
* Confidence intervals
* Probability distributions
* Hypothesis testing
* Regression modeling

## Conclusion

The analysis shows that M-rated games have higher average global sales, but sales performance is also influenced by other factors such as genre, platform, franchise popularity, and market reach. Therefore, violent content alone should not be considered the only reason for higher sales.
