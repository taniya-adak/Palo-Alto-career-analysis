"""
Career Progression and Promotion Gap Analysis for Retention Optimization
Palo Alto Networks - Full Analytical Pipeline

Steps:
1. Load & inspect data
2. Feature engineering (promotion gap ratio, role stagnation, training intensity, manager stability)
3. Preprocessing (encode categoricals, scale numerics, outlier handling)
4. Career path clustering (KMeans + Hierarchical validation)
5. Cluster interpretation & labeling
6. Promotion Gap Risk Scoring (Low/Medium/High)
7. Retention Opportunity Identification
8. Export enriched dataset for BI tools (Power BI / Streamlit)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv('/mnt/user-data/uploads/Palo_Alto_Networks.csv')
print(f"Loaded dataset: {df.shape[0]} employees, {df.shape[1]} columns")

# ---------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------
# Avoid division by zero for employees with YearsAtCompany == 0
df['YearsAtCompany_safe'] = df['YearsAtCompany'].replace(0, 1)

df['PromotionGapRatio'] = df['YearsSinceLastPromotion'] / df['YearsAtCompany_safe']
df['RoleStagnationIndex'] = df['YearsInCurrentRole'] / df['YearsAtCompany_safe']
df['TrainingIntensityScore'] = df['TrainingTimesLastYear'] / df['YearsAtCompany_safe']

# Manager Stability Indicator: proportion of tenure spent with current manager
df['ManagerStabilityIndicator'] = df['YearsWithCurrManager'] / df['YearsAtCompany_safe']

# Clip ratios at 1.0 (can exceed slightly due to rounding in source data)
for col in ['PromotionGapRatio', 'RoleStagnationIndex', 'ManagerStabilityIndicator']:
    df[col] = df[col].clip(0, 1)

print("\nEngineered features summary:")
print(df[['PromotionGapRatio', 'RoleStagnationIndex', 'TrainingIntensityScore',
          'ManagerStabilityIndicator']].describe().round(3))

# ---------------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------------
cluster_features = [
    'PromotionGapRatio', 'RoleStagnationIndex', 'TrainingIntensityScore',
    'ManagerStabilityIndicator', 'YearsAtCompany', 'YearsSinceLastPromotion',
    'YearsInCurrentRole', 'TrainingTimesLastYear', 'NumCompaniesWorked',
    'JobLevel', 'JobInvolvement', 'PerformanceRating'
]

X = df[cluster_features].copy()

# Outlier handling: cap at 1st/99th percentile (retain late-career edge cases but reduce distortion)
for col in ['YearsAtCompany', 'YearsSinceLastPromotion', 'YearsInCurrentRole', 'NumCompaniesWorked']:
    lower, upper = X[col].quantile(0.01), X[col].quantile(0.99)
    X[col] = X[col].clip(lower, upper)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------
# 4. CAREER PATH CLUSTERING
# ---------------------------------------------------------------
# Test silhouette across k for reference, but fix k=5 to match the 5 business
# archetypes named in the project spec (Fast-track / Stable / Early-career /
# Promotion-stalled / High-risk stagnation). A coarser k scores higher on
# silhouette but collapses distinct risk groups together, which isn't usable
# for this business question.
sil_scores = {}
for k in range(3, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, labels)

print(f"\nSilhouette scores by k (reference only): { {k: round(v,3) for k,v in sil_scores.items()} }")
best_k = 5
print(f"Using k = {best_k} to match the 5 career archetypes in the project spec")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['CareerCluster_ID'] = kmeans.fit_predict(X_scaled)

# Hierarchical clustering as validation
hier = AgglomerativeClustering(n_clusters=best_k)
df['HierCluster_ID'] = hier.fit_predict(X_scaled)

# Agreement between methods (Adjusted Rand Index)
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(df['CareerCluster_ID'], df['HierCluster_ID'])
print(f"KMeans vs Hierarchical agreement (ARI): {ari:.3f}")

# ---------------------------------------------------------------
# 5. CLUSTER INTERPRETATION & LABELING
# ---------------------------------------------------------------
cluster_profile = df.groupby('CareerCluster_ID')[cluster_features].mean().round(2)
print("\nCluster profiles (mean values):")
print(cluster_profile)

# Rank-based labeling: compares clusters RELATIVE to each other (not fixed
# absolute thresholds), since with 5 clusters the natural groupings shift.
profile = cluster_profile.copy()
profile['StagnationScore'] = profile['PromotionGapRatio'] + profile['RoleStagnationIndex']

# Identify the early-career cluster first (lowest tenure) and set it aside,
# since low tenure explains low promotion gap for reasons unrelated to risk.
early_career_id = profile['YearsAtCompany'].idxmin()

remaining = profile.drop(index=early_career_id)
stagnation_rank = remaining['StagnationScore'].rank(ascending=False)  # 1 = most stagnant
training_rank = remaining['TrainingIntensityScore'].rank(ascending=True)  # 1 = least trained

labels = {early_career_id: 'Early-Career Explorer'}
sorted_by_stagnation = stagnation_rank.sort_values().index.tolist()  # most -> least stagnant

# Most stagnant cluster overall = Promotion-Stalled
labels[sorted_by_stagnation[0]] = 'Promotion-Stalled'
# Second-most stagnant, especially if also low on training = High-Risk Stagnation
labels[sorted_by_stagnation[1]] = 'High-Risk Stagnation'
# Least stagnant remaining cluster with strongest JobLevel/PerformanceRating = Fast-Track Performer
fast_track_id = remaining.drop(index=[sorted_by_stagnation[0], sorted_by_stagnation[1]])['JobLevel'].idxmax()
labels[fast_track_id] = 'Fast-Track Performer'

# Whatever's left = Stable Long-Term Contributor
for cid in profile.index:
    if cid not in labels:
        labels[cid] = 'Stable Long-Term Contributor'

cluster_labels = pd.Series(labels).sort_index()
print("\nCluster labels assigned:")
print(cluster_labels)

df['CareerCluster'] = df['CareerCluster_ID'].map(cluster_labels)

# ---------------------------------------------------------------
# 6. PROMOTION GAP RISK SCORING
# ---------------------------------------------------------------
def promotion_gap_score(row):
    score = 0
    score += row['PromotionGapRatio'] * 40
    score += row['RoleStagnationIndex'] * 30
    score += (1 - min(row['TrainingIntensityScore'], 1)) * 15
    score += (row['ManagerStabilityIndicator']) * 5  # long single-manager tenure w/ no growth = risk
    score += min(row['NumCompaniesWorked'], 5) * 2  # restless history adds slight risk context
    return round(score, 1)

df['PromotionGapRiskScore'] = df.apply(promotion_gap_score, axis=1)

# Bucket into Low/Medium/High using tertiles
q1, q2 = df['PromotionGapRiskScore'].quantile([0.33, 0.66])
df['PromotionGapRiskLevel'] = pd.cut(
    df['PromotionGapRiskScore'],
    bins=[-1, q1, q2, 200],
    labels=['Low', 'Medium', 'High']
)

print(f"\nPromotion Gap Risk Level distribution:")
print(df['PromotionGapRiskLevel'].value_counts())

# ---------------------------------------------------------------
# 7. RETENTION OPPORTUNITY IDENTIFICATION
# ---------------------------------------------------------------
# Retention Opportunity = NOT already flagged as high attrition risk (Attrition==1 means they already left,
# so opportunity pool = current employees, Attrition==0), showing stagnation signals, not yet critically dissatisfied
df['RetentionOpportunity'] = np.where(
    (df['Attrition'] == 0) &
    (df['PromotionGapRiskLevel'].isin(['Medium', 'High'])) &
    (df['JobSatisfaction'] >= 2),  # not already at rock-bottom satisfaction
    'Yes', 'No'
)

# Retention Opportunity Index (0-100): higher = more valuable/urgent to act on
def retention_opportunity_index(row):
    if row['RetentionOpportunity'] == 'No':
        return 0
    idx = row['PromotionGapRiskScore'] * 0.6
    idx += (row['JobLevel'] * 5)  # more senior = higher cost of losing them
    idx += (row['PerformanceRating'] * 5)
    idx -= (4 - row['JobSatisfaction']) * 3  # already unhappy = less "opportunity", more "emergency"
    return round(max(idx, 0), 1)

df['RetentionOpportunityIndex'] = df.apply(retention_opportunity_index, axis=1)

print(f"\nRetention Opportunity pool: {(df['RetentionOpportunity']=='Yes').sum()} employees "
      f"({(df['RetentionOpportunity']=='Yes').mean()*100:.1f}%)")

# ---------------------------------------------------------------
# 8. TRAINING NEED INDICATOR & MANAGER STABILITY IMPACT
# ---------------------------------------------------------------
df['TrainingNeedIndicator'] = np.where(
    df['TrainingIntensityScore'] < df['TrainingIntensityScore'].median(), 'High Need', 'Adequate'
)

# Manager Stability Impact: correlate long manager tenure with stagnation
df['ManagerStabilityImpact'] = np.where(
    (df['ManagerStabilityIndicator'] > 0.7) & (df['RoleStagnationIndex'] > 0.5),
    'Possible Growth Barrier',
    'Neutral/Positive'
)

# ---------------------------------------------------------------
# 9. EXPORT ENRICHED DATASET
# ---------------------------------------------------------------
output_cols = list(df.columns.drop(['YearsAtCompany_safe', 'HierCluster_ID', 'CareerCluster_ID']))
df_export = df[output_cols]

df_export.to_csv('/home/claude/palo_alto_enriched.csv', index=False)
print(f"\nExported enriched dataset: {df_export.shape[0]} rows, {df_export.shape[1]} columns")
print("Saved to palo_alto_enriched.csv")

# Save cluster profile summary too
cluster_summary = df.groupby('CareerCluster').agg(
    Employee_Count=('CareerCluster', 'count'),
    Avg_PromotionGapRatio=('PromotionGapRatio', 'mean'),
    Avg_RoleStagnation=('RoleStagnationIndex', 'mean'),
    Avg_TrainingIntensity=('TrainingIntensityScore', 'mean'),
    Avg_TenureYears=('YearsAtCompany', 'mean'),
    Attrition_Rate=('Attrition', 'mean'),
    Retention_Opportunities=('RetentionOpportunity', lambda x: (x == 'Yes').sum())
).round(3)
cluster_summary.to_csv('/home/claude/cluster_summary.csv')
print("\nCluster Summary:")
print(cluster_summary)
