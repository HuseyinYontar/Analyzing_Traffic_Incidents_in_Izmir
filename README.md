# Analyzing Traffic Incidents in Izmir

A comprehensive data-driven analysis of traffic incident patterns in Izmir's metropolitan area, examining **21,161 incident records** spanning December 2021 to September 2025.

[![Paper](https://img.shields.io/badge/Read_the_Paper-PDF-red?style=for-the-badge&logo=adobeacrobatreader)](https://ieeexplore.ieee.org/document/11537079)

## Abstract

Traffic incidents in Izmir impose severe life-threatening risks and substantial economic costs on citizens and municipal authorities. This study addresses the gap in comprehensive data-driven analyses by applying classification, time-series regression, clustering, and ensemble learning methods to identify spatiotemporal patterns and develop predictive models for incident severity and daily counts.

### Key Findings

- **Spatial Concentration**: Konak district accounts for more incidents than all other districts combined
- **Critical Hotspots**: Three bridge overpasses are responsible for 13% of recorded incidents
- **Temporal Patterns**: Late-night incidents exhibit 88% injury/fatal proportions compared to 22-30% during peak hours
- **Predictive Accuracy**: 74% accuracy in severity classification and ~7% error in monthly forecasts

## Dataset

The dataset was obtained from the [Izmir Metropolitan Municipality Open Data Portal](https://ulasav.csb.gov.tr/dataset/35-izmir-ili-arizali-kazali-arac-verileri), authored by Izmir Transportation Center Directorate (IZUM).

| Attribute | Type | Description |
|-----------|------|-------------|
| Date | Interval | 2021-12-01 to 2025-09-28 |
| Street | Nominal | 183 distinct streets |
| Direction | Nominal | 157 distinct directions |
| Location | Nominal | 2,032 distinct locations |
| Incident Type | Nominal | 30 distinct types |
| Incident Time | Interval | 00:00 - 23:59 |
| Intervention Time | Interval | Response timestamp |

### Metropolitan Districts Covered
Çiğli, Karşıyaka, Bayraklı, Bornova, Konak, Buca, Karabağlar, Gaziemir, Balçova, and Narlıdere.

## Methodology

### 1. Exploratory Data Analysis
- Incident distribution by district and street
- Hourly incident patterns (working vs. non-working days)
- Severity distribution across time intervals
- Correlation heatmaps between time intervals and incident types

### 2. Supervised Learning - Classification
Predicting life-threatening vs. non-life-threatening incidents using:
- **MLP Neural Network** (Best: 73.82% accuracy)
- Random Forest (72.12%)
- Logistic Regression (72.53%)
- AdaBoost (70.66%)
- Decision Tree (70.50%)
- K-Nearest Neighbors (65.15%)

### 3. Supervised Learning - Regression
Monthly incident volume forecasting using:
- **Neural Network (MLP)**: MAPE 6-20% depending on location
- **Time-series models**: SARIMA, ARIMA variants
- Linear Regression

### 4. Clustering Analysis
Unsupervised analysis of districts and streets using:
- K-Means clustering
- DBSCAN
- Hierarchical clustering (single/complete linkage)

## Project Structure

```
├── dataCleaning/          # Data preprocessing scripts
├── plots/                 # Generated visualizations
├── classification/        # Classification model implementations
├── regression/            # Time-series regression models
├── clustering/            # Clustering analysis scripts
├── ensembleLearning/      # Ensemble model implementations
├── heatmap/               # Correlation analysis
├── pca/                   # Principal Component Analysis
├── associationMining/     # Association rule mining
├── attribute_selection/   # Feature selection analysis
├── time_series_analysis/  # Time series decomposition
├── location/              # Geographic analysis
└── weatherData/           # External weather data
```

## Weather Data

Daily weather observations for *Gaziemir* and *Çiğli* were collected from [Wunderground](https://www.wunderground.com/) for supplementary analysis.
- [Gaziemir Weather Data](https://www.wunderground.com/history/daily/tr/gaziemir/LTBJ/date/2025-9-29)
- [Çiğli Weather Data](https://www.wunderground.com/history/daily/tr/%C3%A7i%C4%9Fli/LTBL/date/2025-9-29)

## Top Incident Hotspots

| Location | Incidents |
|----------|-----------|
| Hilal Bridge Overpass (Mürselpaşa Blvd.) | 954 |
| Tepecik Bridge Overpass (Yeşildere Ave.) | 949 |
| DGM Bridge Overpass (Mürselpaşa Blvd.) | 886 |
| Free Zone Underpass (Akçay Ave.) | 521 |
| Gaziemir Underpass (Akçay Ave.) | 497 |
| After Naldöken Bridge (Anadolu Ave.) | 484 |

## Conclusions

- **Targeted Infrastructure**: Bridge overpasses require priority safety interventions
- **Time-specific Enforcement**: Late-night periods need enhanced monitoring despite lower volume
- **Risk-tiered Resources**: Cluster-based resource allocation enables efficient deployment
- **Predictive Planning**: Accurate forecasting supports proactive incident management

## Authors

İzmir University of Economics
- Efe Korhan Uz - Industrial Engineering
- Hüseyin Atacan Akgün - Software Engineering  
- Hüseyin Yontar - Software Engineering
- Supervisor: Alper Demir
