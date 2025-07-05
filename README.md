# hybrid-recommender-system
Hybrid Recommender System for E-Commerce (Collaborative + Content-Based)

# 🛒 Hybrid Recommender System for E-Commerce (Collaborative + Content-Based)

A recommendation engine built for e-commerce platforms that blends **Collaborative Filtering** and **Content-Based Filtering** to provide personalized and scalable product recommendations.

---

## 🚀 Project Highlights

- ✅ End-to-end hybrid recommender built using Python and pandas  
- ✅ Combines user-product interactions with product metadata  
- ✅ Item-based Collaborative Filtering via user-rating matrix  
- ✅ Content-Based Filtering via TF-IDF on product features  
- ✅ Hybrid scoring using weighted fusion of both methods  
- ✅ Handles cold-starts and missing data scenarios  
- ✅ Modular & scalable: can be plugged into a real e-commerce backend

---

## 🔧 Problem Statement

Recommend relevant products to users on an e-commerce platform by leveraging:

- Past purchase or interaction history (collaborative filtering)  
- Product metadata (brand, category, price, description, etc.)

---

## 🧱 Architecture Overview

```
          +---------------------+
          | interactions.csv    |
          | products.csv        |
          +---------------------+
                    |
             ┌──────▼──────┐
             │ Data Loader │
             └──────┬──────┘
        +-----------+-----------+
        |                       |
┌───────▼───────┐       ┌───────▼────────┐
│ Collaborative │       │ Content-Based  │
│ Filtering     │       │ Filtering (TFIDF)
└───────┬───────┘       └───────┬────────┘
        |                       |
        +-----------+-----------+
                    |
             ┌──────▼──────┐
             │  Hybrid     │   ← α·CF + (1−α)·CBF
             │  Recommender│
             └──────┬──────┘
                    ▼
              🔍 Top-N Products
```
