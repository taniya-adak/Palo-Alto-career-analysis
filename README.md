# Career Progression and Promotion Gap Analysis for Retention Optimization

A career-trajectory intelligence project for Palo Alto Networks that goes beyond traditional attrition prediction — instead of only flagging *who* might leave, it identifies *why* employees may eventually disengage, using promotion gaps and career stagnation patterns.

## Problem

Most attrition models predict who is likely to leave, but not the structural career issues driving that decision — long promotion gaps, role stagnation, limited training, or weak manager continuity. This project builds a data-driven view of career progression across the workforce to catch these issues early, before an employee disengages.

## Approach

1. **Feature Engineering** — built four custom metrics from raw HR data:
   - Promotion Gap Ratio
   - Role Stagnation Index
   - Training Intensity Score
   - Manager Stability Indicator

2. **Career Path Clustering** — K-Means clustering (k=5), validated against Hierarchical clustering, segmenting 1,470 employees into five career archetypes:
   - Fast-Track Performer
   - Stable Long-Term Contributor
   - Early-Career Explorer
   - Promotion-Stalled
   - High-Risk Stagnation

3. **Promotion Gap Risk Scoring** — a weighted score (0–100) combining promotion gap, role stagnation, training deficit, manager tenure concentration, and job-hopping history, bucketed into Low / Medium / High risk.

4. **Retention Opportunity Identification** — flags current employees showing stagnation signals before disengagement, each mapped to a suggested action (Promotion Review, Role Rotation, Training & Skill Development, Stretch Assignment, or Mentorship Check-in).

## Key Finding

The longest-tenured employee segment (18.6 years average tenure) has the **lowest** attrition rate (9.4%) but the **second-highest** role stagnation — a population invisible to standard attrition models. These employees aren't leaving, but 132 of them show real stagnation signals worth proactive intervention.

## Repository Contents

| File | Description |
|---|---|
| `pipeline.py` | Full Python pipeline: feature engineering, clustering, risk scoring, retention flagging |
| `data/palo_alto_enriched.csv` | Original dataset enriched with all engineered features, cluster assignments, and risk scores |
| `cluster_summary.csv` | Aggregated KPIs by career cluster |
| `PaloAlto_Career_Dashboard.pbix` | Interactive Power BI dashboard covering 4 modules: Career Path Clustering, Promotion Gap Monitor, Retention Opportunity Panel, and Managerial Insight |

## Tools Used

- **Python** (pandas, scikit-learn) — feature engineering, K-Means & Hierarchical clustering, risk scoring
- **Power BI** — interactive dashboard with slicers, threshold sliders, and a DAX-calculated column for suggested actions

## Dashboard

The Power BI dashboard includes:
- Career cluster distribution and summary stats
- Promotion gap risk monitoring by job role
- A sorted, filterable retention opportunity panel
- Manager tenure vs. role stagnation insights

## Author

Taniya Adak
