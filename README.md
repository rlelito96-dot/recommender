# Product Recommender System

Work-in-progress project exploring collaborative filtering recommendations
using cosine similarity, built to learn fundamentals of ML (pandas, numpy,
scikit-learn) alongside backend development skills.

## Status
🚧 In progress — currently implementing model evaluation (precision@k).

## Dataset
[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — 
UCI Machine Learning Repository.

## What's implemented
- Data cleaning and exploratory analysis
- Product-product similarity matrix using cosine similarity (scikit-learn)
- Basic recommendation function returning top-N similar products

## What's next
- Train/test evaluation (leave-one-out methodology, precision@k)
- FastAPI endpoint to serve recommendations
- MLflow experiment tracking
- Docker + AWS deployment

## Tech stack
Python, pandas, numpy, scikit-learn, Jupyter