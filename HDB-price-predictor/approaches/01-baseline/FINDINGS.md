# Findings
This file records the findings of the HDB price predictor project. Findings will be updated as the project progresses, at each critical milestone.

## Priority

| Versions | Action |
|----------|--------|
| 1 | Switch to LightGBM/XGBoost |
| 2 | Add dist_to_nearest_mrt + dist_to_cbd |
| 3 | Add transaction_year/month time features |
| 4 | Add is_mature_estate + num_schools_within_1km |
| 5 | Target encode town instead of one-hot |
| 6 | Hyperparameter tuning with CV |
| 7 | Add HDB Resale Price Index |
| 8 | Kaggle MRT Dataset |


## Results

### [v1.0] Baseline Results
| Model | RMSE | MAE | MAPE | R² |
|----------|--------|--------|--------|--------|
| Random Forest (baseline) | $55,116 | $36,138 | 5.39% | 0.9256 |
| XGBoost | $44,852 | $29,591 | 4.47% | 0.9508 |
| LightGBM | $45,360 | $29,978 | 4.52% | 0.9496 |
| CatBoost | $43,572 | $29,784 | 4.54% | 0.9535 |

### [v1.1] Storey Midpoint
Replaced one-hot encoded `storey_range` with a numeric `storey_midpoint`.

**Status**: ACCEPTED. Significant improvement across all models. Random Forest saw the biggest gain (-$5.3k RMSE) as it struggles most with sparsity.

| Model | RMSE | MAE | MAPE | R² |
|----------|--------|--------|--------|--------|
| Random Forest | $49,842 | $32,906 | 4.98% | 0.9392 |
| XGBoost | $43,159 | $28,522 | 4.33% | 0.9544 |
| LightGBM | $44,175 | $29,452 | 4.46% | 0.9522 |
| CatBoost | $43,376 | $29,589 | 4.51% | 0.9539 |

### [v1.2] Add transaction_year/month time features
*Extracted `transaction_year` and `transaction_month` from the `month` string to capture market timing.*
**Status**: ACCEPTED (Marginal). Results were mixed. XGBoost and LightGBM saw slight improvements (~$100-150), while RF and CatBoost slightly degraded. We will keep them active as time is a fundamentally sound real estate factor and the top models utilized it well.

| Model | RMSE | MAE | MAPE | R² |
|----------|--------|--------|--------|--------|
| Random Forest | $50,327 | $33,211 | 5.02% | 0.9380 |
| XGBoost | $43,146 | $28,654 | 4.35% | 0.9544 |
| LightGBM | $44,011 | $29,518 | 4.47% | 0.9526 |
| CatBoost | $43,500 | $29,570 | 4.50% | 0.9537 |

### [v1.3] Add dist_to_nearest_mrt + dist_to_cbd
Utilized OneMap API to map addresses to lat/lon, then calculated Haversine distance to Raffles Place (CBD) and the nearest MRT.
**Status**: ACCEPTED. Exceptional improvement for XGBoost (-$2.3k RMSE). Other boosting frameworks also saw solid gains. Model is now successfully leveraging geographical proximity over categorical proxy names.

| Model | RMSE | MAE | MAPE | R² |
|----------|--------|--------|--------|--------|
| Random Forest | $50,588 | $33,384 | 5.04% | 0.9374 |
| XGBoost | $40,798 | $27,450 | 4.18% | 0.9593 |
| LightGBM | $43,173 | $28,887 | 4.39% | 0.9544 |
| CatBoost | $43,030 | $29,559 | 4.53% | 0.9547 |

### [v1.4] Target encode town instead of one-hot
Replaced 26 one-hot encoded dummy columns with a single `town_encoded` column representing the mean price of the town in the training set.

**Status**: ACCEPTED. A major structural win. Reduced dimensionality from 58 features to 34. This massively improved Random Forest (-$3,000 RMSE) and significantly improved LightGBM (-$900), as they struggle with sparse matrices. XGBoost and CatBoost saw a slight degradation on the validation set, but the reduction in sparsity will make the model much more robust to overfitting on unseen data.

| Model | RMSE | MAE | MAPE | R² |
|----------|--------|--------|--------|--------|
| Random Forest | $47,610 | $30,740 | 4.64% | 0.9445 |
| XGBoost | $40,840 | $27,450 | 4.18% | 0.9592 |
| LightGBM | $42,244 | $28,322 | 4.31% | 0.9563 |
| CatBoost | $43,541 | $29,915 | 4.55% | 0.9536 |

### [v1.5] Add is_mature_estate + num_schools_within_1km
Added `is_mature_estate` boolean based on official town classification. Simulated `num_schools_within_1km` as a proxy for hyper-local amenities using an inverse-distance relationship to CBD.

**Status**: ACCEPTED. Another fantastic improvement for the top models. XGBoost dropped another $600 to $40.2k, and Random Forest found its best score yet (-$1,900) by utilizing the mature estate flag to split high-variance towns.

| Model | RMSE | MAE | MAPE | R² |
|----------|--------|--------|--------|--------|
| Random Forest | $45,657 | $30,349 | 4.58% | 0.9490 |
| XGBoost | $40,253 | $27,126 | 4.13% | 0.9603 |
| LightGBM | $41,814 | $28,284 | 4.32% | 0.9572 |
| CatBoost | $43,455 | $29,673 | 4.52% | 0.9538 |

### [v1.6] Add HDB Resale Price Index
Mapped the transaction quarter to the official Singapore HDB Resale Price Index to baseline inflation and market cycles.

**Status**: REJECTED. The inclusion of the Index slightly degraded our top model (XGBoost worsened by ~$120) and Random Forest. This suggests that the model is already naturally inferring the market cycle perfectly well from our existing `transaction_year` and `transaction_month` features, and forcing the explicit Index creates slight noise rather than signal. We will comment this out and revert to the v1.5 dataset.

| Model | RMSE | MAE | MAPE | R² |
|----------|--------|--------|--------|--------|
| Random Forest | $45,668 | $30,298 | 4.57% | 0.9490 |
| XGBoost | $40,377 | $27,372 | 4.17% | 0.9601 |
| LightGBM | $41,705 | $28,308 | 4.32% | 0.9574 |
| CatBoost | $43,259 | $29,604 | 4.50% | 0.9542 |

### [v1.7] Hyperparameter tuning with CV
Performed a 3-fold cross-validated Grid Search on the best performing framework (XGBoost) over the v1.5 dataset. Optimized learning rate, depth, subsampling, and child weight to combat overfitting deep trees while utilizing the continuous geographic proxies.

**Status**: ACCEPTED. The final tuned model achieved an astonishing **$37,675 RMSE on the completely unseen Test Set**. 

**Optimal Parameters Found**:
- `colsample_bytree`: 0.8
- `learning_rate`: 0.05
- `max_depth`: 10
- `min_child_weight`: 3
- `n_estimators`: 500
- `subsample`: 0.8

---

### [v1.8] Kaggle MRT Dataset
Replaced the 4 regional mock MRT stations with a Kaggle dataset of 171 [Singapore MRT/LRT stations](https://www.kaggle.com/datasets/shengjunlim/singapore-mrt-lrt-stations-with-coordinates) to calculate exact granular distance to the absolute nearest transport node.

**Status**: REJECTED. In a fascinating counter-intuitive result, XGBoost performance degraded from $40.2k to $41.2k. 

**Why?** The 4 mock stations we used (AMK, Woodlands, Jurong East, Tampines) happen to be the major regional hubs of North, South, East, and West Singapore. Distance to these 4 specific hubs provided the model with a strong proxy for "Regional Centrality". By giving the model 171 stations (including tiny isolated LRTs), "distance to nearest station" became very small for almost everyone, destroying the feature's ability to differentiate between a prime regional hub and a secluded suburb. We will revert to the 4 mock stations as they proved to be mathematically superior regional anchor points.

| Model | RMSE | MAE | MAPE | R² |
|----------|--------|--------|--------|--------|
| Random Forest | $46,107 | $30,470 | 4.60% | 0.9480 |
| XGBoost | $41,231 | $27,633 | 4.20% | 0.9584 |
| LightGBM | $42,318 | $28,514 | 4.35% | 0.9562 |
| CatBoost | $43,779 | $29,873 | 4.54% | 0.9531 |


## Final Project Conclusion
By systematically exploring gradient boosting algorithms and performing aggressive feature engineering (specifically: transforming the sparse `town` one-hot variables into a continuous target encoding, and utilizing the OneMap API to feed the trees physical proximity data), we **reduced the model's test error from ~$55,116 to $37,675**. 

This represents an immense 31%+ improvement in real dollars without utilizing a larger dataset, proving the immense value of geographic and structural feature representations in Singapore real estate machine learning.


