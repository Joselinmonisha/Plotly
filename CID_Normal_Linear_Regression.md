# CID — Normal Linear Regression
**Concept · Implement · Demonstrate**

---

## C — Concept

### Introduction

Normal linear regression models a **linear relationship** between a continuous **response variable** (Y) and one or more **predictor variables** (X). The objective is to estimate how changes in X relate to expected changes in Y, and to quantify the uncertainty around those estimates.

The modelling process follows **four sequential stages**:

| Stage | Name | Purpose |
|-------|------|---------|
| 1 | Model Building | Define variables and structure |
| 2 | Model Fitting | Estimate parameters |
| 3 | Model Adequacy | Check assumptions and performance |
| 4 | Model Diagnostics | Identify problematic observations |

---

### Stage 1 — Model Building

The first stage is exploratory and structural:

- **Metadata review** — Understand the dataset: source, size, variable types, missing values.
- **Variable selection** — Identify the response variable (Y) and the regressor(s) (X).
- **Nature of variables** — Determine whether variables are continuous, discrete, or categorical.
- **Transformation decisions** — Assess whether log, square root, or other transformations are needed to linearise relationships or stabilise variance.

---

### Stage 2 — Model Fitting

Once the model is specified, unknown parameters (intercept β₀, slopes β₁, β₂, …) are estimated.

#### Parameter Estimation
Two main methods are used:
- **Least Squares Estimation (LSE)** — Minimises the sum of squared residuals. No distributional assumption required.
- **Maximum Likelihood Estimation (MLE)** — Assumes normally distributed errors; produces the same estimates as LSE under normality.

#### Standard Error
The **standard error** was computed for each Estimated parameters. Smaller values indicate more precise estimates.

 

#### Tests of Significance

| Test | Purpose |
|------|---------|
| **ANOVA (F-test)** | Tests whether the overall model (all predictors together) is statistically significant |
| **t-test** | Tests whether each individual parameter is significant, holding others constant |

#### Confidence Intervals
- If the interval **includes zero** (bounds have opposite signs) → fail to reject H₀ → the predictor is **not statistically significant**
- If the interval **excludes zero** (bounds have the same sign) → reject H₀ → the predictor **is significant**, and the sign of the bounds indicates direction of effect

---

### Stage 3 — Model Adequacy

Before trusting results, the four regression assumptions must be verified.

#### The Four Assumptions

| Assumption | Description |
|------------|-------------|
| **Linearity** | The relationship between Y and each X is linear |
| **Homoscedasticity** | The error term has constant variance across all fitted values |
| **Independence** | Errors are uncorrelated with each other |
| **Normality** | Errors are normally distributed |

#### Diagnostic Plots

| Plot | What It Checks |
|------|---------------|
| Residuals vs Fitted | Linearity and homoscedasticity |
| Normal Q-Q Plot | Normality of residuals |
| Histogram of Residuals | Distribution shape of errors |
| Actual vs Predicted | Overall predictive accuracy |

#### Formal Tests
- **Shapiro-Wilk test** — Tests normality of residuals (preferred for small samples)
- **Kolmogorov-Smirnov test** — Tests normality (better for larger samples)

#### Performance Metrics

| Metric | Description |
|--------|-------------|
| **R²** | Proportion of variance in Y explained by the model |
| **Adjusted R²** | R² penalised for number of predictors; fairer for multi-predictor models |
| **MSE / RMSE** | Mean squared / root mean squared prediction error |
| **PRESS Statistic** | Leave-one-out cross-validation error; measures predictive performance |
| **Lack of Fit** | Formal test for whether the model form is appropriate |

---

### Stage 4 — Model Diagnostics

Identifies observations that violate assumptions or have undue influence on the model.

#### Types of Problematic Observations

- **Outlier** — An observation with an unusually large residual; it deviates from the pattern the model captures.
- **Leverage point** — An observation extreme in predictor space (X). Does *not* necessarily affect model coefficients but can inflate R² and distort standard errors.
- **Influential point** — An observation that substantially changes the model coefficients when included vs. excluded.

#### Detection Metrics

| Metric | What It Detects |
|--------|----------------|
| **Leverage (hᵢᵢ)** | Extremeness of an observation in X-space |
| **Cook's D** | Overall influence on all fitted values |
| **DFFITS** | Change in fitted value for observation i when it is removed |
| **DFBETAS** | Change in each individual coefficient when observation i is removed |
| **COVRATIO** | Effect of observation i on the precision (covariance matrix) of estimates |

---

## I — Implement

Implementation focuses on handling **categorical predictors** in linear regression across three platforms.

---

### R

R's default behaviour is well-suited to categorical variables.

- Categorical variables are handled via **factoring** — wrapping a variable with `factor()` automatically triggers dummy encoding.
- R selects the **first level** (alphabetically) as the reference category by default.
- Coefficients for other levels represent the difference from this reference.

```r
model <- lm(Y ~ X + factor(Category), data = df)
summary(model)
```

---

### Python

Three libraries offer different approaches:

#### 1. `statsmodels` — Manual Encoding
```python
import statsmodels.api as sm
import pandas as pd

df = pd.get_dummies(df, columns=['Category'], drop_first=True)  # dummy variable method
X = sm.add_constant(df[['X', 'Category_B', 'Category_C']])
model = sm.OLS(df['Y'], X).fit()
print(model.summary())
```

#### 2. `scikit-learn` — Encoding Options

**Default: LabelEncoder** (assigns integer codes — not recommended for nominal categories)
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Category_encoded'] = le.fit_transform(df['Category'])
```

**Preferred: Dummy Variable (OneHotEncoder)**
```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(drop='first')  # drop first to avoid multicollinearity
```

Use `scipy.stats.f_oneway` or `scipy.stats.pearsonr` for supplementary significance testing.

#### 3. `Patsy` — R-style syntax in Python

Patsy is the most intuitive option for those familiar with R. It automatically handles encoding, choosing one level as the reference and comparing all others to it.

```python
import statsmodels.formula.api as smf

model = smf.ols('Y ~ X + C(Category)', data=df).fit()
print(model.summary())
```

- `C(Category)` tells Patsy to treat the variable as categorical.
- The reference level is chosen automatically (first level) and can be overridden with `C(Category, Treatment('LevelName'))`.

### SAS

SAS handles categorical predictors through primary procedure `PROC REG` requiring manual encoding 

#### `PROC REG` — Manual Dummy Encoding

`PROC REG` does not natively support categorical variables, so dummy variables must be created in a preceding `DATA` step. One level is dropped to serve as the reference category and avoid perfect multicollinearity.

```sas
/* Step 1: Create dummy variables manually */
DATA work.encoded;
    SET mydata;
    Cat_B = (Category = 'B');  /* 1 if B, 0 otherwise */
    Cat_C = (Category = 'C');  /* 1 if C, 0 otherwise */
    /* Category = 'A' is the reference level (omitted) */
RUN;

/* Step 2: Fit the model */
PROC REG DATA = work.encoded;
    MODEL Y = X Cat_B Cat_C / CLB;  /* CLB prints confidence limits for coefficients */
RUN;
QUIT;
```

- The omitted level (`Category = 'A'`) is the **reference category** — all coefficients are interpreted relative to it.





## Summary & Comparison

| Feature | R | Python (statsmodels) | Python (scikit-learn) | Python (Patsy) | SAS |
|---|---|---|---|---|---|
| **Encoding** | Inbuild (factor) | Manual / Patsy | Inbuild (LabelEncoder) / Manual (OneHotEncoder) | Inbuild (formula) | Manual (PROC REG) |
| **Reference category** | First level | First column dropped | First category dropped | First alphabetical | Reference option |
| **Formula interface** | `lm(Y ~ X)` | `smf.ols('Y ~ X')` | None (pipeline-based) | `dmatrices('Y ~ X')` | `MODEL Y = X;` |
| **Statistical output** | Full summary | Full summary | Metrics only | Full summary via statsmodels | Full |
| **Underlying engine** | Base R | SciPy | SciPy | SciPy | SAS IML |


---
---

## D — Demonstrate
Demonstrating Categorical Predictor Handling in different platforms

## **R** :
```

bw <- read.csv("C:/Users/Admin/Downloads/BirthWt.csv")

# Convert categorical variables to factors
bw$sex <- factor(bw$sex)
bw$ht   <- factor(bw$ht)   # hypertension: yes/no

# Fit linear model
# bweight ~ maternal age + hypertension + gestational weeks + sex
model <- lm(bweight ~ matage + ht + gestwks + sex, data = bw)

# Model summary
summary(model)

# Diagnostic plots
par(mfrow = c(2, 2))
plot(model)


# ── Model Diagnostics ─────────────────────────────────────────────────────────

n   <- nrow(bw)
p   <- length(coef(model))
dfb <- dfbetas(model)

lev_thresh     <- 2 * p / n
cooksD_thresh  <- 4 / n
dffits_thresh  <- 2 * sqrt(p / n)
dfbetas_thresh <- 2 / sqrt(n)
covratio_upper <- 1 + 3 * p / n
covratio_lower <- 1 - 3 * p / n

# --- Flag influential observations ---
cat("=== High Leverage ===\n")
print(which(hatvalues(model) > lev_thresh))

cat("\n=== Influential by Cook's D ===\n")
print(which(cooks.distance(model) > cooksD_thresh))

cat("\n=== Influential by DFFITS ===\n")
print(which(abs(dffits(model)) > dffits_thresh))

cat("\n=== Influential by DFBETAS ===\n")
for (j in 1:ncol(dfb)) {
  flagged <- which(abs(dfb[, j]) > dfbetas_thresh)
  if (length(flagged) > 0)
    cat(sprintf("  Coefficient '%s': obs %s\n",
                colnames(dfb)[j], paste(flagged, collapse = ", ")))
}

cat("\n=== Influential by COVRATIO ===\n")
print(which(covratio(model) < covratio_lower | covratio(model) > covratio_upper))


# --- Diagnostic Plots: separate pages to avoid layout overflow ---

# Page 1: Leverage, Cook's D, DFFITS, COVRATIO
dev.new()
par(mfrow = c(2, 2))

plot(hatvalues(model), type = "h",
     main = "Leverage", ylab = "Hat Values", xlab = "Observation", col = "steelblue")
abline(h = lev_thresh, col = "red", lty = 2)

plot(cooks.distance(model), type = "h",
     main = "Cook's Distance", ylab = "Cook's D", xlab = "Observation", col = "steelblue")
abline(h = cooksD_thresh, col = "red", lty = 2)

plot(dffits(model), type = "h",
     main = "DFFITS", ylab = "DFFITS", xlab = "Observation", col = "steelblue")
abline(h =  dffits_thresh, col = "red", lty = 2)
abline(h = -dffits_thresh, col = "red", lty = 2)

plot(covratio(model), type = "h",
     main = "COVRATIO", ylab = "COVRATIO", xlab = "Observation", col = "steelblue")
abline(h = covratio_upper, col = "red", lty = 2)
abline(h = covratio_lower, col = "red", lty = 2)

par(mfrow = c(1, 1))

# Page 2: DFBETAS (one plot per coefficient)
n_coef <- ncol(dfb)
dev.new()
par(mfrow = c(ceiling(n_coef / 2), 2))

for (j in 1:n_coef) {
  plot(dfb[, j], type = "h",
       main = paste("DFBETAS:", colnames(dfb)[j]),
       ylab = "DFBETAS", xlab = "Observation", col = "steelblue")
  abline(h =  dfbetas_thresh, col = "red", lty = 2)
  abline(h = -dfbetas_thresh, col = "red", lty = 2)
}

par(mfrow = c(1, 1))

```

## **Python (statsmodels)**

### **Manual Encoding**

```
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# 1. LOAD & ENCODE
# ══════════════════════════════════════════════════════════════
df = pd.read_csv(r"C:\Users\Admin\Downloads\BirthWt.csv")

df['sex_male'] = (df['sex'] == 'male').astype(int)   # female=0 male=1
df['ht_yes']   = (df['ht']  == 'yes').astype(int)    # no=0    yes=1

X = sm.add_constant(df[['matage','ht_yes','gestwks','sex_male']])
y = df['bweight']

# ══════════════════════════════════════════════════════════════
# 2. FIT MODEL
# ══════════════════════════════════════════════════════════════
model     = sm.OLS(y, X).fit()
influence = OLSInfluence(model)
print(model.summary())

n, k      = len(y), X.shape[1]
fitted    = model.fittedvalues.values
residuals = model.resid.values

# ══════════════════════════════════════════════════════════════
# 3. DIAGNOSTICS
# ══════════════════════════════════════════════════════════════
hat_diag     = influence.hat_matrix_diag
cooks_d      = influence.cooks_distance[0]
dffits_val   = influence.dffits[0]
dfbetas_val  = influence.dfbetas
covratio_val = influence.cov_ratio
r_student    = influence.resid_studentized_external

cut_lev   = 2 * k / n
cut_cook  = 1.0
cut_dff   = 2 * np.sqrt(k / n)
cut_dfb   = 2 / np.sqrt(n)
cut_cov_u = 1 + 3 * k / n
cut_cov_l = 1 - 3 * k / n

obs = np.arange(n)

print(f"\n{'='*50}")
print(f"  CUTOFFS  (n={n}, k={k})")
print(f"{'='*50}")
print(f"  Leverage  > {cut_lev:.4f}")
print(f"  Cook's D  > {cut_cook:.2f}")
print(f"  DFFITS    > {cut_dff:.4f}")
print(f"  DFBETAS   > {cut_dfb:.4f}")
print(f"  COVRATIO outside [{cut_cov_l:.4f}, {cut_cov_u:.4f}]")

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def split(x, y_arr, flags):
    return (x[~flags], y_arr[~flags],
            x[ flags], y_arr[ flags])

def stems(x, y_arr, flags, cnorm, cflag):
    lx, ly = [], []
    for xi, yi in zip(x, y_arr):
        lx += [xi, xi, None]
        ly += [0,  yi, None]
    return go.Scatter(
        x=lx, y=ly, mode='lines',
        line=dict(color=cnorm, width=0.7),
        hoverinfo='skip', showlegend=False)

THEME = 'plotly_white'
FLAG  = '#E53935'

# ══════════════════════════════════════════════════════════════
# 4. MODEL ADEQUACY  (2 × 2)
# ══════════════════════════════════════════════════════════════
fa = make_subplots(
    rows=2, cols=2,
    subplot_titles=['Residuals vs Fitted','Normal Q-Q Plot',
                    'Histogram of Residuals','Actual vs Predicted'],
    vertical_spacing=0.18, horizontal_spacing=0.12)

# 4a. Residuals vs Fitted
fa.add_trace(go.Scatter(
    x=fitted, y=residuals, mode='markers',
    marker=dict(color='steelblue', size=5, opacity=0.5),
    hovertemplate='Fitted=%{x:.0f}<br>Resid=%{y:.0f}<extra></extra>',
    name='Residuals'), row=1, col=1)
fa.add_hline(y=0, line=dict(color='red', dash='dash', width=1.5),
             row=1, col=1)
fa.update_xaxes(title_text='Fitted Values', row=1, col=1)
fa.update_yaxes(title_text='Residuals',     row=1, col=1)

# 4b. Normal Q-Q
(osm, osr), (slope, intercept, _) = stats.probplot(residuals)
fa.add_trace(go.Scatter(
    x=osm, y=osr, mode='markers',
    marker=dict(color='steelblue', size=4, opacity=0.5),
    hovertemplate='Theoretical=%{x:.3f}<br>Sample=%{y:.3f}<extra></extra>',
    name='Q-Q'), row=1, col=2)
fa.add_trace(go.Scatter(
    x=[min(osm), max(osm)],
    y=[slope*min(osm)+intercept, slope*max(osm)+intercept],
    mode='lines', line=dict(color='red', width=1.8),
    showlegend=False), row=1, col=2)
fa.update_xaxes(title_text='Theoretical Quantiles', row=1, col=2)
fa.update_yaxes(title_text='Sample Quantiles',      row=1, col=2)

# 4c. Histogram
xr    = np.linspace(residuals.min(), residuals.max(), 300)
pdf_y = stats.norm.pdf(xr, residuals.mean(), residuals.std())
fa.add_trace(go.Histogram(
    x=residuals, nbinsx=30, histnorm='probability density',
    marker=dict(color='seagreen',
                line=dict(color='white', width=0.5)),
    opacity=0.8, name='Histogram',
    hovertemplate='Resid=%{x:.0f}<br>Density=%{y:.5f}<extra></extra>'),
    row=2, col=1)
fa.add_trace(go.Scatter(
    x=xr, y=pdf_y, mode='lines',
    line=dict(color='crimson', width=1.8),
    name='Normal curve'), row=2, col=1)
fa.update_xaxes(title_text='Residuals', row=2, col=1)
fa.update_yaxes(title_text='Density',   row=2, col=1)

# 4d. Actual vs Predicted
mn, mx = y.min(), y.max()
fa.add_trace(go.Scatter(
    x=fitted, y=y.values, mode='markers',
    marker=dict(color='darkorange', size=5, opacity=0.5),
    hovertemplate='Pred=%{x:.0f}<br>Actual=%{y:.0f}<extra></extra>',
    name='Actual vs Pred'), row=2, col=2)
fa.add_trace(go.Scatter(
    x=[mn, mx], y=[mn, mx], mode='lines',
    line=dict(color='red', dash='dash', width=1.8),
    name='Perfect fit'), row=2, col=2)
fa.update_xaxes(title_text='Predicted (ŷ)', row=2, col=2)
fa.update_yaxes(title_text='Actual (y)',    row=2, col=2)

fa.update_layout(
    template=THEME, height=650,
    title=dict(
        text=(f'<b>Model Adequacy – Birth Weight</b><br>'
              f'<sup>R²={model.rsquared:.4f}  '
              f'Adj R²={model.rsquared_adj:.4f}  '
              f'RMSE={np.sqrt(model.mse_resid):.1f} g  '
              f'n={n}</sup>'),
        x=0.5, font=dict(size=14)),
    showlegend=False,
    margin=dict(l=60, r=40, t=90, b=60))
fa.show()

# ══════════════════════════════════════════════════════════════
# 5. MODEL DIAGNOSTICS  (2 × 3, 6th cell hidden)
# ══════════════════════════════════════════════════════════════
fd = make_subplots(
    rows=2, cols=3,
    subplot_titles=[
        '① Leverage (h_ii)',
        "② Cook's Distance (D_i)",
        '③ |DFFITS|',
        '④ COVRATIO',
        '⑤ DFBETAS – gestwks',
        ''],
    vertical_spacing=0.18, horizontal_spacing=0.10)

# ── flags ──────────────────────────────────────────────────────
fl   = hat_diag > cut_lev
fc   = cooks_d  > cut_cook
fd_  = np.abs(dffits_val) > cut_dff
fcv  = (covratio_val > cut_cov_u) | (covratio_val < cut_cov_l)
dfbg = dfbetas_val[:, 3]          # gestwks column
fdb  = np.abs(dfbg) > cut_dfb

# ── Plot 1 : Leverage ──────────────────────────────────────────
xn,yn,xf,yf = split(obs, hat_diag, fl)
fd.add_trace(stems(obs, hat_diag, fl, 'steelblue', FLAG), row=1, col=1)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='steelblue', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>h_ii=%{y:.5f}<extra></extra>'),
    row=1, col=1)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f'Leverage>{cut_lev:.4f}',
    hovertemplate='Obs %{x}<br>h_ii=%{y:.5f} ⚠<extra></extra>'),
    row=1, col=1)
fd.add_hline(y=cut_lev, line=dict(color=FLAG, dash='dash', width=1.5),
             annotation_text=f'Cutoff={cut_lev:.4f}',
             annotation_font_color=FLAG, row=1, col=1)
fd.update_xaxes(title_text='Obs Index', row=1, col=1)
fd.update_yaxes(title_text='h_ii',      row=1, col=1)

# ── Plot 2 : Cook's D ──────────────────────────────────────────
xn,yn,xf,yf = split(obs, cooks_d, fc)
fd.add_trace(stems(obs, cooks_d, fc, 'darkorange', FLAG), row=1, col=2)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='darkorange', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>D=%{y:.6f}<extra></extra>'),
    row=1, col=2)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f"Cook's D>{cut_cook}",
    hovertemplate='Obs %{x}<br>D=%{y:.6f} ⚠<extra></extra>'),
    row=1, col=2)
fd.add_hline(y=cut_cook, line=dict(color=FLAG, dash='dash', width=1.5),
             annotation_text=f'Cutoff={cut_cook}',
             annotation_font_color=FLAG, row=1, col=2)
fd.update_xaxes(title_text='Obs Index', row=1, col=2)
fd.update_yaxes(title_text='D_i',       row=1, col=2)

# ── Plot 3 : DFFITS ────────────────────────────────────────────
abs_dff      = np.abs(dffits_val)
xn,yn,xf,yf = split(obs, abs_dff, fd_)
fd.add_trace(stems(obs, abs_dff, fd_, '#7B1FA2', FLAG), row=1, col=3)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='#7B1FA2', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>|DFFITS|=%{y:.4f}<extra></extra>'),
    row=1, col=3)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f'|DFFITS|>{cut_dff:.4f}',
    hovertemplate='Obs %{x}<br>|DFFITS|=%{y:.4f} ⚠<extra></extra>'),
    row=1, col=3)
fd.add_hline(y=cut_dff, line=dict(color=FLAG, dash='dash', width=1.5),
             annotation_text=f'Cutoff={cut_dff:.4f}',
             annotation_font_color=FLAG, row=1, col=3)
fd.update_xaxes(title_text='Obs Index',  row=1, col=3)
fd.update_yaxes(title_text='|DFFITS_i|', row=1, col=3)

# ── Plot 4 : COVRATIO ──────────────────────────────────────────
xn,yn,xf,yf = split(obs, covratio_val, fcv)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='#00897B', size=5, opacity=0.65),
    showlegend=False,
    hovertemplate='Obs %{x}<br>COVRATIO=%{y:.4f}<extra></extra>'),
    row=2, col=1)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name='COVRATIO flagged',
    hovertemplate='Obs %{x}<br>COVRATIO=%{y:.4f} ⚠<extra></extra>'),
    row=2, col=1)
fd.add_hline(y=cut_cov_u, line=dict(color=FLAG, dash='dash', width=1.5),
             annotation_text=f'Upper={cut_cov_u:.4f}',
             annotation_font_color=FLAG, row=2, col=1)
fd.add_hline(y=cut_cov_l, line=dict(color='royalblue', dash='dash', width=1.5),
             annotation_text=f'Lower={cut_cov_l:.4f}',
             annotation_font_color='royalblue',
             annotation_position='bottom right', row=2, col=1)
fd.add_hline(y=1.0, line=dict(color='gray', dash='dot', width=1),
             row=2, col=1)
fd.update_xaxes(title_text='Obs Index',  row=2, col=1)
fd.update_yaxes(title_text='COVRATIO_i', row=2, col=1)

# ── Plot 5 : DFBETAS gestwks ───────────────────────────────────
xn,yn,xf,yf = split(obs, dfbg, fdb)
fd.add_trace(stems(obs, dfbg, fdb, '#2E7D32', FLAG), row=2, col=2)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='#2E7D32', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>DFBETAS=%{y:.4f}<extra></extra>'),
    row=2, col=2)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f'|DFBETAS|>{cut_dfb:.4f}',
    hovertemplate='Obs %{x}<br>DFBETAS=%{y:.4f} ⚠<extra></extra>'),
    row=2, col=2)
fd.add_hline(y= cut_dfb, line=dict(color=FLAG, dash='dash', width=1.5),
             annotation_text=f'+Cutoff={cut_dfb:.4f}',
             annotation_font_color=FLAG, row=2, col=2)
fd.add_hline(y=-cut_dfb, line=dict(color=FLAG, dash='dash', width=1.5),
             annotation_text=f'-Cutoff={-cut_dfb:.4f}',
             annotation_font_color=FLAG,
             annotation_position='bottom right', row=2, col=2)
fd.add_hline(y=0, line=dict(color='gray', dash='dot', width=1),
             row=2, col=2)
fd.update_xaxes(title_text='Obs Index', row=2, col=2)
fd.update_yaxes(title_text='DFBETAS',   row=2, col=2)

```

### **Manual encoding using Dummy Variable:**

```
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# 1. LOAD & DUMMY ENCODE
# ══════════════════════════════════════════════════════════════
df = pd.read_csv(r"C:\Users\Admin\Downloads\BirthWt.csv")

# Dummy variable encoding  (drop_first=True avoids multicollinearity)
# sex  → sex_male   (reference = female, dropped)
# ht   → ht_yes     (reference = no,     dropped)
dummies = pd.get_dummies(df[['sex','ht']], drop_first=True)
dummies = dummies.astype(int)          # bool → int  (fixes sklearn/sm issue)
dummies.columns = ['sex_male','ht_yes']

df = pd.concat([df, dummies], axis=1)

print("Dummy encoding:")
print(df[['sex','sex_male','ht','ht_yes']].drop_duplicates()
        .sort_values('sex_male').to_string(index=False))

# ══════════════════════════════════════════════════════════════
# 2. FIT OLS MODEL  (statsmodels)
# ══════════════════════════════════════════════════════════════
X = sm.add_constant(df[['matage','ht_yes','gestwks','sex_male']])
y = df['bweight']

model     = sm.OLS(y, X).fit()
influence = OLSInfluence(model)
print(model.summary())

n, k      = len(y), X.shape[1]
fitted    = model.fittedvalues.values
residuals = model.resid.values

# ══════════════════════════════════════════════════════════════
# 3. EXTRACT DIAGNOSTICS
# ══════════════════════════════════════════════════════════════
hat_diag     = influence.hat_matrix_diag
cooks_d      = influence.cooks_distance[0]
dffits_val   = influence.dffits[0]
dfbetas_val  = influence.dfbetas          # shape (n, k)
covratio_val = influence.cov_ratio
r_student    = influence.resid_studentized_external

# cutoffs
cut_lev   = 2 * k / n
cut_cook  = 1.0
cut_dff   = 2 * np.sqrt(k / n)
cut_dfb   = 2 / np.sqrt(n)
cut_cov_u = 1 + 3 * k / n
cut_cov_l = 1 - 3 * k / n

obs = np.arange(n)

print(f"\n{'='*50}")
print(f"  CUTOFFS  (n={n}, k={k})")
print(f"{'='*50}")
print(f"  Leverage  > {cut_lev:.4f}")
print(f"  Cook's D  > {cut_cook:.2f}")
print(f"  DFFITS    > {cut_dff:.4f}")
print(f"  DFBETAS   > {cut_dfb:.4f}")
print(f"  COVRATIO outside [{cut_cov_l:.4f}, {cut_cov_u:.4f}]")

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
THEME = 'plotly_white'
FLAG  = '#E53935'

def split(x, y_arr, flags):
    """Split arrays into normal / flagged."""
    return (x[~flags], y_arr[~flags],
            x[ flags], y_arr[ flags])

def stems(x, y_arr, cnorm):
    """Vertical stem lines as a single Scatter trace."""
    lx, ly = [], []
    for xi, yi in zip(x, y_arr):
        lx += [xi, xi, None]
        ly += [0,  yi, None]
    return go.Scatter(
        x=lx, y=ly, mode='lines',
        line=dict(color=cnorm, width=0.7),
        hoverinfo='skip', showlegend=False)

# ══════════════════════════════════════════════════════════════
# 4. MODEL ADEQUACY  (2 × 2)
# ══════════════════════════════════════════════════════════════
fa = make_subplots(
    rows=2, cols=2,
    subplot_titles=['Residuals vs Fitted', 'Normal Q-Q Plot',
                    'Histogram of Residuals','Actual vs Predicted'],
    vertical_spacing=0.18, horizontal_spacing=0.12)

# ── 4a. Residuals vs Fitted ───────────────────────────────────
fa.add_trace(go.Scatter(
    x=fitted, y=residuals, mode='markers',
    marker=dict(color='steelblue', size=5, opacity=0.5),
    hovertemplate='Fitted=%{x:.0f}<br>Resid=%{y:.0f}<extra></extra>',
    name='Residuals'), row=1, col=1)
fa.add_hline(y=0, line=dict(color='red', dash='dash', width=1.5),
             row=1, col=1)
fa.update_xaxes(title_text='Fitted Values', row=1, col=1)
fa.update_yaxes(title_text='Residuals',     row=1, col=1)

# ── 4b. Normal Q-Q ───────────────────────────────────────────
(osm, osr), (slope, intercept, _) = stats.probplot(residuals)
fa.add_trace(go.Scatter(
    x=osm, y=osr, mode='markers',
    marker=dict(color='steelblue', size=4, opacity=0.5),
    hovertemplate='Theoretical=%{x:.3f}<br>Sample=%{y:.3f}<extra></extra>',
    name='Q-Q'), row=1, col=2)
fa.add_trace(go.Scatter(
    x=[min(osm), max(osm)],
    y=[slope*min(osm)+intercept, slope*max(osm)+intercept],
    mode='lines', line=dict(color='red', width=1.8),
    showlegend=False), row=1, col=2)
fa.update_xaxes(title_text='Theoretical Quantiles', row=1, col=2)
fa.update_yaxes(title_text='Sample Quantiles',      row=1, col=2)

# ── 4c. Histogram ─────────────────────────────────────────────
xr    = np.linspace(residuals.min(), residuals.max(), 300)
pdf_y = stats.norm.pdf(xr, residuals.mean(), residuals.std())
fa.add_trace(go.Histogram(
    x=residuals, nbinsx=30, histnorm='probability density',
    marker=dict(color='seagreen', line=dict(color='white', width=0.5)),
    opacity=0.8, name='Histogram',
    hovertemplate='Resid=%{x:.0f}<br>Density=%{y:.5f}<extra></extra>'),
    row=2, col=1)
fa.add_trace(go.Scatter(
    x=xr, y=pdf_y, mode='lines',
    line=dict(color='crimson', width=1.8),
    name='Normal curve'), row=2, col=1)
fa.update_xaxes(title_text='Residuals', row=2, col=1)
fa.update_yaxes(title_text='Density',   row=2, col=1)

# ── 4d. Actual vs Predicted ───────────────────────────────────
mn, mx = y.min(), y.max()
fa.add_trace(go.Scatter(
    x=fitted, y=y.values, mode='markers',
    marker=dict(color='darkorange', size=5, opacity=0.5),
    hovertemplate='Pred=%{x:.0f}<br>Actual=%{y:.0f}<extra></extra>',
    name='Actual vs Pred'), row=2, col=2)
fa.add_trace(go.Scatter(
    x=[mn, mx], y=[mn, mx], mode='lines',
    line=dict(color='red', dash='dash', width=1.8),
    name='Perfect fit'), row=2, col=2)
fa.update_xaxes(title_text='Predicted (ŷ)', row=2, col=2)
fa.update_yaxes(title_text='Actual (y)',    row=2, col=2)

fa.update_layout(
    template=THEME, height=650, showlegend=False,
    title=dict(
        text=(f'<b>Model Adequacy – Birth Weight</b><br>'
              f'<sup>Encoding: sex_male (ref=female)  '
              f'ht_yes (ref=no)  |  '
              f'R²={model.rsquared:.4f}  '
              f'Adj R²={model.rsquared_adj:.4f}  '
              f'RMSE={np.sqrt(model.mse_resid):.1f} g  '
              f'n={n}</sup>'),
        x=0.5, font=dict(size=13)),
    margin=dict(l=60, r=40, t=90, b=60))
fa.show()

# ══════════════════════════════════════════════════════════════
# 5. MODEL DIAGNOSTICS  (2 × 3, 6th hidden)
# ══════════════════════════════════════════════════════════════

# flags
fl   = hat_diag > cut_lev
fc   = cooks_d  > cut_cook
fdf  = np.abs(dffits_val) > cut_dff
fcv  = (covratio_val > cut_cov_u) | (covratio_val < cut_cov_l)
dfbg = dfbetas_val[:, 3]          # gestwks column (index 3)
fdb  = np.abs(dfbg) > cut_dfb

fd = make_subplots(
    rows=2, cols=3,
    subplot_titles=[
        '① Leverage (h_ii)',
        "② Cook's Distance (D_i)",
        '③ |DFFITS|',
        '④ COVRATIO',
        '⑤ DFBETAS – gestwks',
        ''],
    vertical_spacing=0.18, horizontal_spacing=0.10)

# ── Plot 1 : Leverage ─────────────────────────────────────────
xn, yn, xf, yf = split(obs, hat_diag, fl)
fd.add_trace(stems(obs, hat_diag, 'steelblue'),   row=1, col=1)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='steelblue', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>h_ii=%{y:.5f}<extra></extra>'),
    row=1, col=1)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f'Leverage > {cut_lev:.4f}',
    hovertemplate='Obs %{x}<br>h_ii=%{y:.5f} ⚠<extra></extra>'),
    row=1, col=1)
fd.add_hline(y=cut_lev,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'Cutoff={cut_lev:.4f}',
    annotation_font_color=FLAG, row=1, col=1)
fd.update_xaxes(title_text='Obs Index', row=1, col=1)
fd.update_yaxes(title_text='h_ii',      row=1, col=1)

# ── Plot 2 : Cook's D ─────────────────────────────────────────
xn, yn, xf, yf = split(obs, cooks_d, fc)
fd.add_trace(stems(obs, cooks_d, 'darkorange'),   row=1, col=2)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='darkorange', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>D=%{y:.6f}<extra></extra>'),
    row=1, col=2)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f"Cook's D > {cut_cook}",
    hovertemplate='Obs %{x}<br>D=%{y:.6f} ⚠<extra></extra>'),
    row=1, col=2)
fd.add_hline(y=cut_cook,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'Cutoff={cut_cook}',
    annotation_font_color=FLAG, row=1, col=2)
fd.update_xaxes(title_text='Obs Index', row=1, col=2)
fd.update_yaxes(title_text='D_i',       row=1, col=2)

# ── Plot 3 : DFFITS ───────────────────────────────────────────
abs_dff        = np.abs(dffits_val)
xn, yn, xf, yf = split(obs, abs_dff, fdf)
fd.add_trace(stems(obs, abs_dff, '#7B1FA2'),      row=1, col=3)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='#7B1FA2', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>|DFFITS|=%{y:.4f}<extra></extra>'),
    row=1, col=3)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f'|DFFITS| > {cut_dff:.4f}',
    hovertemplate='Obs %{x}<br>|DFFITS|=%{y:.4f} ⚠<extra></extra>'),
    row=1, col=3)
fd.add_hline(y=cut_dff,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'Cutoff={cut_dff:.4f}',
    annotation_font_color=FLAG, row=1, col=3)
fd.update_xaxes(title_text='Obs Index',  row=1, col=3)
fd.update_yaxes(title_text='|DFFITS_i|', row=1, col=3)

# ── Plot 4 : COVRATIO ─────────────────────────────────────────
xn, yn, xf, yf = split(obs, covratio_val, fcv)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='#00897B', size=5, opacity=0.65),
    showlegend=False,
    hovertemplate='Obs %{x}<br>COVRATIO=%{y:.4f}<extra></extra>'),
    row=2, col=1)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name='COVRATIO flagged',
    hovertemplate='Obs %{x}<br>COVRATIO=%{y:.4f} ⚠<extra></extra>'),
    row=2, col=1)
fd.add_hline(y=cut_cov_u,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'Upper={cut_cov_u:.4f}',
    annotation_font_color=FLAG, row=2, col=1)
fd.add_hline(y=cut_cov_l,
    line=dict(color='royalblue', dash='dash', width=1.5),
    annotation_text=f'Lower={cut_cov_l:.4f}',
    annotation_font_color='royalblue',
    annotation_position='bottom right', row=2, col=1)
fd.add_hline(y=1.0,
    line=dict(color='gray', dash='dot', width=1),
    row=2, col=1)
fd.update_xaxes(title_text='Obs Index',  row=2, col=1)
fd.update_yaxes(title_text='COVRATIO_i', row=2, col=1)

# ── Plot 5 : DFBETAS – gestwks ────────────────────────────────
xn, yn, xf, yf = split(obs, dfbg, fdb)
fd.add_trace(stems(obs, dfbg, '#2E7D32'),         row=2, col=2)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='#2E7D32', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>DFBETAS=%{y:.4f}<extra></extra>'),
    row=2, col=2)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f'|DFBETAS| > {cut_dfb:.4f}',
    hovertemplate='Obs %{x}<br>DFBETAS=%{y:.4f} ⚠<extra></extra>'),
    row=2, col=2)
fd.add_hline(y= cut_dfb,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'+Cutoff={cut_dfb:.4f}',
    annotation_font_color=FLAG, row=2, col=2)
fd.add_hline(y=-cut_dfb,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'-Cutoff={-cut_dfb:.4f}',
    annotation_font_color=FLAG,
    annotation_position='bottom right', row=2, col=2)
fd.add_hline(y=0,
    line=dict(color='gray', dash='dot', width=1),
    row=2, col=2)
fd.update_xaxes(title_text='Obs Index', row=2, col=2)
fd.update_yaxes(title_text='DFBETAS',   row=2, col=2)


```

## **Python (scikit-learn)** 

### **Inbuilt (LabelEncoder)**
```
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# 1. LOAD & ENCODE  (LabelEncoder)
# ══════════════════════════════════════════════════════════════
df = pd.read_csv(r"C:\Users\Admin\Downloads\BirthWt.csv")

le          = LabelEncoder()
df['sex_enc'] = le.fit_transform(df['sex'])   # female=0  male=1
df['ht_enc']  = le.fit_transform(df['ht'])    # no=0      yes=1

print("Encoding:")
print(df[['sex','sex_enc','ht','ht_enc']].drop_duplicates()
        .sort_values('sex_enc').to_string(index=False))

# ══════════════════════════════════════════════════════════════
# 2. FIT MODEL  (sklearn)
# ══════════════════════════════════════════════════════════════
FEATURES = ['matage','ht_enc','gestwks','sex_enc']
X_raw = df[FEATURES].values.astype(float)
y     = df['bweight'].values.astype(float)
n, p  = X_raw.shape

sk    = LinearRegression().fit(X_raw, y)
fitted    = sk.predict(X_raw)
residuals = y - fitted

print("Intercept :", round(sk.intercept_, 4))
for f, c in zip(FEATURES, sk.coef_):
    print(f"  {f:12s}: {c:.4f}")
    
r2     = r2_score(y, fitted)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
rmse   = np.sqrt(mean_squared_error(y, fitted))

print(f"\nsklearn R²={r2:.4f}  Adj R²={adj_r2:.4f}  RMSE={rmse:.1f} g")

# ══════════════════════════════════════════════════════════════
# 3. DIAGNOSTICS  (statsmodels — same coefficients as sklearn)
# ══════════════════════════════════════════════════════════════
Xsm       = sm.add_constant(X_raw)
sm_model  = sm.OLS(y, Xsm).fit()
influence = OLSInfluence(sm_model)
k         = Xsm.shape[1]           # intercept + 4 predictors = 5

hat_diag     = influence.hat_matrix_diag
cooks_d      = influence.cooks_distance[0]
dffits_val   = influence.dffits[0]
dfbetas_val  = influence.dfbetas
covratio_val = influence.cov_ratio
r_student    = influence.resid_studentized_external

cut_lev   = 2 * k / n
cut_cook  = 1.0
cut_dff   = 2 * np.sqrt(k / n)
cut_dfb   = 2 / np.sqrt(n)
cut_cov_u = 1 + 3 * k / n
cut_cov_l = 1 - 3 * k / n
obs       = np.arange(n)

print(f"\n{'='*50}")
print(f"  CUTOFFS  (n={n}, k={k})")
print(f"{'='*50}")
print(f"  Leverage  > {cut_lev:.4f}")
print(f"  Cook's D  > {cut_cook:.2f}")
print(f"  DFFITS    > {cut_dff:.4f}")
print(f"  DFBETAS   > {cut_dfb:.4f}")
print(f"  COVRATIO outside [{cut_cov_l:.4f}, {cut_cov_u:.4f}]")

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
THEME = 'plotly_white'
FLAG  = '#E53935'

def split(x, y_arr, flags):
    return (x[~flags], y_arr[~flags],
            x[ flags], y_arr[ flags])

def stems(x, y_arr, cnorm):
    lx, ly = [], []
    for xi, yi in zip(x, y_arr):
        lx += [xi, xi, None]
        ly += [0,  yi, None]
    return go.Scatter(
        x=lx, y=ly, mode='lines',
        line=dict(color=cnorm, width=0.7),
        hoverinfo='skip', showlegend=False)

# ══════════════════════════════════════════════════════════════
# 4. MODEL ADEQUACY  (2 × 2)
# ══════════════════════════════════════════════════════════════
fa = make_subplots(
    rows=2, cols=2,
    subplot_titles=['Residuals vs Fitted','Normal Q-Q Plot',
                    'Histogram of Residuals','Actual vs Predicted'],
    vertical_spacing=0.18, horizontal_spacing=0.12)

# 4a. Residuals vs Fitted
fa.add_trace(go.Scatter(
    x=fitted, y=residuals, mode='markers',
    marker=dict(color='steelblue', size=5, opacity=0.5),
    hovertemplate='Fitted=%{x:.0f}<br>Resid=%{y:.0f}<extra></extra>',
    name='Residuals'), row=1, col=1)
fa.add_hline(y=0,
    line=dict(color='red', dash='dash', width=1.5), row=1, col=1)
fa.update_xaxes(title_text='Fitted Values', row=1, col=1)
fa.update_yaxes(title_text='Residuals',     row=1, col=1)

# 4b. Normal Q-Q
(osm, osr), (slope, intercept, _) = stats.probplot(residuals)
fa.add_trace(go.Scatter(
    x=osm, y=osr, mode='markers',
    marker=dict(color='steelblue', size=4, opacity=0.5),
    hovertemplate='Theoretical=%{x:.3f}<br>Sample=%{y:.3f}<extra></extra>',
    name='Q-Q'), row=1, col=2)
fa.add_trace(go.Scatter(
    x=[min(osm), max(osm)],
    y=[slope*min(osm)+intercept, slope*max(osm)+intercept],
    mode='lines', line=dict(color='red', width=1.8),
    showlegend=False), row=1, col=2)
fa.update_xaxes(title_text='Theoretical Quantiles', row=1, col=2)
fa.update_yaxes(title_text='Sample Quantiles',      row=1, col=2)

# 4c. Histogram
xr    = np.linspace(residuals.min(), residuals.max(), 300)
pdf_y = stats.norm.pdf(xr, residuals.mean(), residuals.std())
fa.add_trace(go.Histogram(
    x=residuals, nbinsx=30, histnorm='probability density',
    marker=dict(color='seagreen',
                line=dict(color='white', width=0.5)),
    opacity=0.8, name='Histogram',
    hovertemplate='Resid=%{x:.0f}<br>Density=%{y:.5f}<extra></extra>'),
    row=2, col=1)
fa.add_trace(go.Scatter(
    x=xr, y=pdf_y, mode='lines',
    line=dict(color='crimson', width=1.8),
    name='Normal curve'), row=2, col=1)
fa.update_xaxes(title_text='Residuals', row=2, col=1)
fa.update_yaxes(title_text='Density',   row=2, col=1)

# 4d. Actual vs Predicted
mn, mx = y.min(), y.max()
fa.add_trace(go.Scatter(
    x=fitted, y=y, mode='markers',
    marker=dict(color='darkorange', size=5, opacity=0.5),
    hovertemplate='Pred=%{x:.0f}<br>Actual=%{y:.0f}<extra></extra>',
    name='Actual vs Pred'), row=2, col=2)
fa.add_trace(go.Scatter(
    x=[mn, mx], y=[mn, mx], mode='lines',
    line=dict(color='red', dash='dash', width=1.8),
    name='Perfect fit'), row=2, col=2)
fa.update_xaxes(title_text='Predicted (ŷ)', row=2, col=2)
fa.update_yaxes(title_text='Actual (y)',    row=2, col=2)

fa.update_layout(
    template=THEME, height=650, showlegend=False,
    title=dict(
        text=(f'<b>Model Adequacy – Birth Weight  (sklearn)</b><br>'
              f'<sup>Encoding: LabelEncoder  |  '
              f'R²={r2:.4f}  Adj R²={adj_r2:.4f}  '
              f'RMSE={rmse:.1f} g  n={n}</sup>'),
        x=0.5, font=dict(size=13)),
    margin=dict(l=60, r=40, t=90, b=60))
fa.show()

# ══════════════════════════════════════════════════════════════
# 5. MODEL DIAGNOSTICS  (2 × 3, 6th hidden)
# ══════════════════════════════════════════════════════════════
fl   = hat_diag > cut_lev
fc   = cooks_d  > cut_cook
fdf  = np.abs(dffits_val) > cut_dff
fcv  = (covratio_val > cut_cov_u) | (covratio_val < cut_cov_l)
dfbg = dfbetas_val[:, 3]           # gestwks column
fdb  = np.abs(dfbg) > cut_dfb

fd = make_subplots(
    rows=2, cols=3,
    subplot_titles=[
        '① Leverage (h_ii)',
        "② Cook's Distance (D_i)",
        '③ |DFFITS|',
        '④ COVRATIO',
        '⑤ DFBETAS – gestwks',
        ''],
    vertical_spacing=0.18, horizontal_spacing=0.10)

# Plot 1 : Leverage
xn,yn,xf,yf = split(obs, hat_diag, fl)
fd.add_trace(stems(obs, hat_diag, 'steelblue'), row=1, col=1)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='steelblue', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>h_ii=%{y:.5f}<extra></extra>'),
    row=1, col=1)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f'Leverage > {cut_lev:.4f}',
    hovertemplate='Obs %{x}<br>h_ii=%{y:.5f} ⚠<extra></extra>'),
    row=1, col=1)
fd.add_hline(y=cut_lev,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'Cutoff={cut_lev:.4f}',
    annotation_font_color=FLAG, row=1, col=1)
fd.update_xaxes(title_text='Obs Index', row=1, col=1)
fd.update_yaxes(title_text='h_ii',      row=1, col=1)

# Plot 2 : Cook's D
xn,yn,xf,yf = split(obs, cooks_d, fc)
fd.add_trace(stems(obs, cooks_d, 'darkorange'), row=1, col=2)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='darkorange', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>D=%{y:.6f}<extra></extra>'),
    row=1, col=2)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f"Cook's D > {cut_cook}",
    hovertemplate='Obs %{x}<br>D=%{y:.6f} ⚠<extra></extra>'),
    row=1, col=2)
fd.add_hline(y=cut_cook,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'Cutoff={cut_cook}',
    annotation_font_color=FLAG, row=1, col=2)
fd.update_xaxes(title_text='Obs Index', row=1, col=2)
fd.update_yaxes(title_text='D_i',       row=1, col=2)

# Plot 3 : DFFITS
abs_dff      = np.abs(dffits_val)
xn,yn,xf,yf = split(obs, abs_dff, fdf)
fd.add_trace(stems(obs, abs_dff, '#7B1FA2'), row=1, col=3)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='#7B1FA2', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>|DFFITS|=%{y:.4f}<extra></extra>'),
    row=1, col=3)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f'|DFFITS| > {cut_dff:.4f}',
    hovertemplate='Obs %{x}<br>|DFFITS|=%{y:.4f} ⚠<extra></extra>'),
    row=1, col=3)
fd.add_hline(y=cut_dff,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'Cutoff={cut_dff:.4f}',
    annotation_font_color=FLAG, row=1, col=3)
fd.update_xaxes(title_text='Obs Index',  row=1, col=3)
fd.update_yaxes(title_text='|DFFITS_i|', row=1, col=3)

# Plot 4 : COVRATIO
xn,yn,xf,yf = split(obs, covratio_val, fcv)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='#00897B', size=5, opacity=0.65),
    showlegend=False,
    hovertemplate='Obs %{x}<br>COVRATIO=%{y:.4f}<extra></extra>'),
    row=2, col=1)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name='COVRATIO flagged',
    hovertemplate='Obs %{x}<br>COVRATIO=%{y:.4f} ⚠<extra></extra>'),
    row=2, col=1)
fd.add_hline(y=cut_cov_u,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'Upper={cut_cov_u:.4f}',
    annotation_font_color=FLAG, row=2, col=1)
fd.add_hline(y=cut_cov_l,
    line=dict(color='royalblue', dash='dash', width=1.5),
    annotation_text=f'Lower={cut_cov_l:.4f}',
    annotation_font_color='royalblue',
    annotation_position='bottom right', row=2, col=1)
fd.add_hline(y=1.0,
    line=dict(color='gray', dash='dot', width=1), row=2, col=1)
fd.update_xaxes(title_text='Obs Index',  row=2, col=1)
fd.update_yaxes(title_text='COVRATIO_i', row=2, col=1)

# Plot 5 : DFBETAS – gestwks
xn,yn,xf,yf = split(obs, dfbg, fdb)
fd.add_trace(stems(obs, dfbg, '#2E7D32'), row=2, col=2)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='#2E7D32', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>DFBETAS=%{y:.4f}<extra></extra>'),
    row=2, col=2)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f'|DFBETAS| > {cut_dfb:.4f}',
    hovertemplate='Obs %{x}<br>DFBETAS=%{y:.4f} ⚠<extra></extra>'),
    row=2, col=2)
fd.add_hline(y= cut_dfb,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'+Cutoff={cut_dfb:.4f}',
    annotation_font_color=FLAG, row=2, col=2)
fd.add_hline(y=-cut_dfb,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'-Cutoff={-cut_dfb:.4f}',
    annotation_font_color=FLAG,
    annotation_position='bottom right', row=2, col=2)
fd.add_hline(y=0,
    line=dict(color='gray', dash='dot', width=1), row=2, col=2)
fd.update_xaxes(title_text='Obs Index', row=2, col=2)
fd.update_yaxes(title_text='DFBETAS',   row=2, col=2)

# Hide 6th cell
fd.update_xaxes(visible=False, row=2, col=3)
fd.update_yaxes(visible=False, row=2, col=3)

fd.update_layout(
    template=THEME, height=700,
    title=dict(
        text=(f'<b>Regression Diagnostics – Birth Weight  (sklearn)</b><br>'
              f'<sup>Encoding: LabelEncoder  |  '
              f'n={n}  k={k}  |  '
              f'Leverage>{cut_lev:.4f}  '
              f"Cook's D>{cut_cook}  "
              f'|DFFITS|>{cut_dff:.4f}  '
              f'|DFBETAS|>{cut_dfb:.4f}</sup>'),
        x=0.5, font=dict(size=13)),
    legend=dict(
        orientation='h', y=-0.12,
        xanchor='center', x=0.5,
        font=dict(size=10)),
    margin=dict(l=60, r=40, t=100, b=80))

fd.show()
```
### **Manual ( Dummy Variable encoding)**

```
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# 1. LOAD & DUMMY ENCODE
# ══════════════════════════════════════════════════════════════
df = pd.read_csv(r"C:\Users\Admin\Downloads\BirthWt.csv")

# Dummy variable encoding
# drop_first=True → female & no are reference (dropped)
dummies = pd.get_dummies(
    df[['sex', 'ht']], drop_first=True
).astype(int)                    # bool → int  (fixes TypeError)
dummies.columns = ['sex_male', 'ht_yes']

df = pd.concat([df, dummies], axis=1)

print("Dummy Encoding:")
print(df[['sex','sex_male','ht','ht_yes']]
        .drop_duplicates()
        .sort_values('sex_male')
        .to_string(index=False))

# ══════════════════════════════════════════════════════════════
# 2. FIT MODEL  (sklearn)
# ══════════════════════════════════════════════════════════════
FEATURES  = ['matage', 'ht_yes', 'gestwks', 'sex_male']
X_raw     = df[FEATURES].values.astype(float)
y         = df['bweight'].values.astype(float)
n, p      = X_raw.shape

sk        = LinearRegression().fit(X_raw, y)
fitted    = sk.predict(X_raw)
residuals = y - fitted

print(f"Intercept : {sk.intercept_:.4f}")
for f, c in zip(FEATURES, sk.coef_):
    print(f"  {f:12s}: {c:.4f}")

r2     = r2_score(y, fitted)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
rmse   = np.sqrt(mean_squared_error(y, fitted))

print(f"\nsklearn  R²={r2:.4f}  Adj R²={adj_r2:.4f}  RMSE={rmse:.1f} g")


# ══════════════════════════════════════════════════════════════
# 3. DIAGNOSTICS  (statsmodels — identical coefficients)
# ══════════════════════════════════════════════════════════════
Xsm       = sm.add_constant(X_raw)
sm_model  = sm.OLS(y, Xsm).fit()
influence = OLSInfluence(sm_model)
k         = Xsm.shape[1]          # intercept + 4 = 5

hat_diag     = influence.hat_matrix_diag
cooks_d      = influence.cooks_distance[0]
dffits_val   = influence.dffits[0]
dfbetas_val  = influence.dfbetas          # (n × k)
covratio_val = influence.cov_ratio
r_student    = influence.resid_studentized_external

cut_lev   = 2 * k / n
cut_cook  = 1.0
cut_dff   = 2 * np.sqrt(k / n)
cut_dfb   = 2 / np.sqrt(n)
cut_cov_u = 1 + 3 * k / n
cut_cov_l = 1 - 3 * k / n
obs       = np.arange(n)

print(f"\n{'='*50}")
print(f"  CUTOFFS  (n={n}, k={k})")
print(f"{'='*50}")
print(f"  Leverage  > {cut_lev:.4f}")
print(f"  Cook's D  > {cut_cook:.2f}")
print(f"  DFFITS    > {cut_dff:.4f}")
print(f"  DFBETAS   > {cut_dfb:.4f}")
print(f"  COVRATIO outside [{cut_cov_l:.4f}, {cut_cov_u:.4f}]")

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
THEME = 'plotly_white'
FLAG  = '#E53935'

def split(x, y_arr, flags):
    return (x[~flags], y_arr[~flags],
            x[ flags], y_arr[ flags])

def stems(x, y_arr, cnorm):
    lx, ly = [], []
    for xi, yi in zip(x, y_arr):
        lx += [xi, xi, None]
        ly += [0,  yi, None]
    return go.Scatter(
        x=lx, y=ly, mode='lines',
        line=dict(color=cnorm, width=0.7),
        hoverinfo='skip', showlegend=False)

# ══════════════════════════════════════════════════════════════
# 4. MODEL ADEQUACY  (2 × 2)
# ══════════════════════════════════════════════════════════════
fa = make_subplots(
    rows=2, cols=2,
    subplot_titles=['Residuals vs Fitted', 'Normal Q-Q Plot',
                    'Histogram of Residuals', 'Actual vs Predicted'],
    vertical_spacing=0.18, horizontal_spacing=0.12)

# 4a. Residuals vs Fitted
fa.add_trace(go.Scatter(
    x=fitted, y=residuals, mode='markers',
    marker=dict(color='steelblue', size=5, opacity=0.5),
    hovertemplate='Fitted=%{x:.0f}<br>Resid=%{y:.0f}<extra></extra>',
    name='Residuals'), row=1, col=1)
fa.add_hline(y=0,
    line=dict(color='red', dash='dash', width=1.5), row=1, col=1)
fa.update_xaxes(title_text='Fitted Values', row=1, col=1)
fa.update_yaxes(title_text='Residuals',     row=1, col=1)

# 4b. Normal Q-Q
(osm, osr), (slope, intercept, _) = stats.probplot(residuals)
fa.add_trace(go.Scatter(
    x=osm, y=osr, mode='markers',
    marker=dict(color='steelblue', size=4, opacity=0.5),
    hovertemplate='Theoretical=%{x:.3f}<br>Sample=%{y:.3f}<extra></extra>',
    name='Q-Q'), row=1, col=2)
fa.add_trace(go.Scatter(
    x=[min(osm), max(osm)],
    y=[slope*min(osm)+intercept, slope*max(osm)+intercept],
    mode='lines', line=dict(color='red', width=1.8),
    showlegend=False), row=1, col=2)
fa.update_xaxes(title_text='Theoretical Quantiles', row=1, col=2)
fa.update_yaxes(title_text='Sample Quantiles',      row=1, col=2)

# 4c. Histogram
xr    = np.linspace(residuals.min(), residuals.max(), 300)
pdf_y = stats.norm.pdf(xr, residuals.mean(), residuals.std())
fa.add_trace(go.Histogram(
    x=residuals, nbinsx=30, histnorm='probability density',
    marker=dict(color='seagreen',
                line=dict(color='white', width=0.5)),
    opacity=0.8, name='Histogram',
    hovertemplate='Resid=%{x:.0f}<br>Density=%{y:.5f}<extra></extra>'),
    row=2, col=1)
fa.add_trace(go.Scatter(
    x=xr, y=pdf_y, mode='lines',
    line=dict(color='crimson', width=1.8),
    name='Normal curve'), row=2, col=1)
fa.update_xaxes(title_text='Residuals', row=2, col=1)
fa.update_yaxes(title_text='Density',   row=2, col=1)

# 4d. Actual vs Predicted
mn, mx = y.min(), y.max()
fa.add_trace(go.Scatter(
    x=fitted, y=y, mode='markers',
    marker=dict(color='darkorange', size=5, opacity=0.5),
    hovertemplate='Pred=%{x:.0f}<br>Actual=%{y:.0f}<extra></extra>',
    name='Actual vs Pred'), row=2, col=2)
fa.add_trace(go.Scatter(
    x=[mn, mx], y=[mn, mx], mode='lines',
    line=dict(color='red', dash='dash', width=1.8),
    name='Perfect fit'), row=2, col=2)
fa.update_xaxes(title_text='Predicted (ŷ)', row=2, col=2)
fa.update_yaxes(title_text='Actual (y)',    row=2, col=2)

fa.update_layout(
    template=THEME, height=650, showlegend=False,
    title=dict(
        text=(f'<b>Model Adequacy – Birth Weight  (sklearn)</b><br>'
              f'<sup>Dummy encoding: sex_male (ref=female)  '
              f'ht_yes (ref=no)  |  '
              f'R²={r2:.4f}  Adj R²={adj_r2:.4f}  '
              f'RMSE={rmse:.1f} g  n={n}</sup>'),
        x=0.5, font=dict(size=13)),
    margin=dict(l=60, r=40, t=90, b=60))
fa.show()

# ══════════════════════════════════════════════════════════════
# 5. MODEL DIAGNOSTICS  (2 × 3, 6th hidden)
# ══════════════════════════════════════════════════════════════
fl   = hat_diag > cut_lev
fc   = cooks_d  > cut_cook
fdf  = np.abs(dffits_val) > cut_dff
fcv  = (covratio_val > cut_cov_u) | (covratio_val < cut_cov_l)
dfbg = dfbetas_val[:, 3]          # gestwks column (index 3)
fdb  = np.abs(dfbg) > cut_dfb

fd = make_subplots(
    rows=2, cols=3,
    subplot_titles=[
        '① Leverage (h_ii)',
        "② Cook's Distance (D_i)",
        '③ |DFFITS|',
        '④ COVRATIO',
        '⑤ DFBETAS – gestwks',
        ''],
    vertical_spacing=0.18, horizontal_spacing=0.10)

# Plot 1 : Leverage
xn, yn, xf, yf = split(obs, hat_diag, fl)
fd.add_trace(stems(obs, hat_diag, 'steelblue'), row=1, col=1)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='steelblue', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>h_ii=%{y:.5f}<extra></extra>'),
    row=1, col=1)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f'Leverage > {cut_lev:.4f}',
    hovertemplate='Obs %{x}<br>h_ii=%{y:.5f} ⚠<extra></extra>'),
    row=1, col=1)
fd.add_hline(y=cut_lev,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'Cutoff={cut_lev:.4f}',
    annotation_font_color=FLAG, row=1, col=1)
fd.update_xaxes(title_text='Obs Index', row=1, col=1)
fd.update_yaxes(title_text='h_ii',      row=1, col=1)

# Plot 2 : Cook's D
xn, yn, xf, yf = split(obs, cooks_d, fc)
fd.add_trace(stems(obs, cooks_d, 'darkorange'), row=1, col=2)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='darkorange', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>D=%{y:.6f}<extra></extra>'),
    row=1, col=2)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f"Cook's D > {cut_cook}",
    hovertemplate='Obs %{x}<br>D=%{y:.6f} ⚠<extra></extra>'),
    row=1, col=2)
fd.add_hline(y=cut_cook,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'Cutoff={cut_cook}',
    annotation_font_color=FLAG, row=1, col=2)
fd.update_xaxes(title_text='Obs Index', row=1, col=2)
fd.update_yaxes(title_text='D_i',       row=1, col=2)

# Plot 3 : DFFITS
abs_dff        = np.abs(dffits_val)
xn, yn, xf, yf = split(obs, abs_dff, fdf)
fd.add_trace(stems(obs, abs_dff, '#7B1FA2'), row=1, col=3)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='#7B1FA2', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>|DFFITS|=%{y:.4f}<extra></extra>'),
    row=1, col=3)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f'|DFFITS| > {cut_dff:.4f}',
    hovertemplate='Obs %{x}<br>|DFFITS|=%{y:.4f} ⚠<extra></extra>'),
    row=1, col=3)
fd.add_hline(y=cut_dff,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'Cutoff={cut_dff:.4f}',
    annotation_font_color=FLAG, row=1, col=3)
fd.update_xaxes(title_text='Obs Index',  row=1, col=3)
fd.update_yaxes(title_text='|DFFITS_i|', row=1, col=3)

# Plot 4 : COVRATIO
xn, yn, xf, yf = split(obs, covratio_val, fcv)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='#00897B', size=5, opacity=0.65),
    showlegend=False,
    hovertemplate='Obs %{x}<br>COVRATIO=%{y:.4f}<extra></extra>'),
    row=2, col=1)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name='COVRATIO flagged',
    hovertemplate='Obs %{x}<br>COVRATIO=%{y:.4f} ⚠<extra></extra>'),
    row=2, col=1)
fd.add_hline(y=cut_cov_u,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'Upper={cut_cov_u:.4f}',
    annotation_font_color=FLAG, row=2, col=1)
fd.add_hline(y=cut_cov_l,
    line=dict(color='royalblue', dash='dash', width=1.5),
    annotation_text=f'Lower={cut_cov_l:.4f}',
    annotation_font_color='royalblue',
    annotation_position='bottom right', row=2, col=1)
fd.add_hline(y=1.0,
    line=dict(color='gray', dash='dot', width=1), row=2, col=1)
fd.update_xaxes(title_text='Obs Index',  row=2, col=1)
fd.update_yaxes(title_text='COVRATIO_i', row=2, col=1)

# Plot 5 : DFBETAS – gestwks
xn, yn, xf, yf = split(obs, dfbg, fdb)
fd.add_trace(stems(obs, dfbg, '#2E7D32'), row=2, col=2)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='#2E7D32', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>DFBETAS=%{y:.4f}<extra></extra>'),
    row=2, col=2)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f'|DFBETAS| > {cut_dfb:.4f}',
    hovertemplate='Obs %{x}<br>DFBETAS=%{y:.4f} ⚠<extra></extra>'),
    row=2, col=2)
fd.add_hline(y= cut_dfb,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'+Cutoff={cut_dfb:.4f}',
    annotation_font_color=FLAG, row=2, col=2)
fd.add_hline(y=-cut_dfb,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'-Cutoff={-cut_dfb:.4f}',
    annotation_font_color=FLAG,
    annotation_position='bottom right', row=2, col=2)
fd.add_hline(y=0,
    line=dict(color='gray', dash='dot', width=1), row=2, col=2)
fd.update_xaxes(title_text='Obs Index', row=2, col=2)
fd.update_yaxes(title_text='DFBETAS',   row=2, col=2)

# Hide 6th cell
fd.update_xaxes(visible=False, row=2, col=3)
fd.update_yaxes(visible=False, row=2, col=3)

fd.update_layout(
    template=THEME, height=700,
    title=dict(
        text=(f'<b>Regression Diagnostics – Birth Weight  (sklearn)</b><br>'
              f'<sup>Dummy encoding: sex_male (ref=female)  '
              f'ht_yes (ref=no)  |  '
              f'n={n}  k={k}  |  '
              f'Leverage>{cut_lev:.4f}  '
              f"Cook's D>{cut_cook}  "
              f'|DFFITS|>{cut_dff:.4f}  '
              f'|DFBETAS|>{cut_dfb:.4f}</sup>'),
        x=0.5, font=dict(size=13)),
    legend=dict(
        orientation='h', y=-0.12,
        xanchor='center', x=0.5,
        font=dict(size=10)),
    margin=dict(l=60, r=40, t=100, b=80))

fd.show()

```

## **Python (Patsy)** 

### **Inbuild**
```
import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════
df = pd.read_csv(r"C:\Users\Admin\Downloads\BirthWt.csv")
print("Shape :", df.shape)
print(df[['sex','ht','bweight']].head(3))

# ══════════════════════════════════════════════════════════════
# 2. PATSY FORMULA  — encodes categorical automatically
#    sex  → sex[T.male]    (reference = female)
#    ht   → ht[T.yes]      (reference = no    )
#    intercept added by default
# ══════════════════════════════════════════════════════════════
FORMULA = 'bweight ~ matage + ht + gestwks + sex'

y_dm, X_dm = patsy.dmatrices(FORMULA, data=df, return_type='dataframe')

y = np.asarray(y_dm).flatten()
X = np.asarray(X_dm)

print("\nPatsy design matrix columns:")
print(list(X_dm.columns))
print(X_dm.head(3).to_string())

# ══════════════════════════════════════════════════════════════
# 3. FIT OLS MODEL
# ══════════════════════════════════════════════════════════════
model     = sm.OLS(y, X).fit()
influence = OLSInfluence(model)
print(model.summary())

n, k      = X.shape
fitted    = model.fittedvalues
residuals = model.resid

# ══════════════════════════════════════════════════════════════
# 4. EXTRACT DIAGNOSTICS
# ══════════════════════════════════════════════════════════════
hat_diag     = influence.hat_matrix_diag
cooks_d      = influence.cooks_distance[0]
dffits_val   = influence.dffits[0]
dfbetas_val  = influence.dfbetas          # (n × k)
covratio_val = influence.cov_ratio
r_student    = influence.resid_studentized_external

cut_lev   = 2 * k / n
cut_cook  = 1.0
cut_dff   = 2 * np.sqrt(k / n)
cut_dfb   = 2 / np.sqrt(n)
cut_cov_u = 1 + 3 * k / n
cut_cov_l = 1 - 3 * k / n
obs       = np.arange(n)

print(f"\n{'='*52}")
print(f"  CUTOFFS  (n={n}, k={k})")
print(f"{'='*52}")
print(f"  Leverage  > {cut_lev:.4f}")
print(f"  Cook's D  > {cut_cook:.2f}")
print(f"  DFFITS    > {cut_dff:.4f}")
print(f"  DFBETAS   > {cut_dfb:.4f}")
print(f"  COVRATIO outside [{cut_cov_l:.4f}, {cut_cov_u:.4f}]")

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
THEME = 'plotly_white'
FLAG  = '#E53935'

def split(x, y_arr, flags):
    return (x[~flags], y_arr[~flags],
            x[ flags], y_arr[ flags])

def stems(x, y_arr, cnorm):
    lx, ly = [], []
    for xi, yi in zip(x, y_arr):
        lx += [xi, xi, None]
        ly += [0,  yi, None]
    return go.Scatter(
        x=lx, y=ly, mode='lines',
        line=dict(color=cnorm, width=0.7),
        hoverinfo='skip', showlegend=False)

# ══════════════════════════════════════════════════════════════
# 5. MODEL ADEQUACY  (2 × 2)
# ══════════════════════════════════════════════════════════════
fa = make_subplots(
    rows=2, cols=2,
    subplot_titles=['Residuals vs Fitted', 'Normal Q-Q Plot',
                    'Histogram of Residuals', 'Actual vs Predicted'],
    vertical_spacing=0.18, horizontal_spacing=0.12)

# 5a. Residuals vs Fitted
fa.add_trace(go.Scatter(
    x=fitted, y=residuals, mode='markers',
    marker=dict(color='steelblue', size=5, opacity=0.5),
    hovertemplate='Fitted=%{x:.0f}<br>Resid=%{y:.0f}<extra></extra>',
    name='Residuals'), row=1, col=1)
fa.add_hline(y=0,
    line=dict(color='red', dash='dash', width=1.5), row=1, col=1)
fa.update_xaxes(title_text='Fitted Values', row=1, col=1)
fa.update_yaxes(title_text='Residuals',     row=1, col=1)

# 5b. Normal Q-Q
(osm, osr), (slope, intercept, _) = stats.probplot(residuals)
fa.add_trace(go.Scatter(
    x=osm, y=osr, mode='markers',
    marker=dict(color='steelblue', size=4, opacity=0.5),
    hovertemplate='Theoretical=%{x:.3f}<br>Sample=%{y:.3f}<extra></extra>',
    name='Q-Q'), row=1, col=2)
fa.add_trace(go.Scatter(
    x=[min(osm), max(osm)],
    y=[slope*min(osm)+intercept, slope*max(osm)+intercept],
    mode='lines', line=dict(color='red', width=1.8),
    showlegend=False), row=1, col=2)
fa.update_xaxes(title_text='Theoretical Quantiles', row=1, col=2)
fa.update_yaxes(title_text='Sample Quantiles',      row=1, col=2)

# 5c. Histogram
xr    = np.linspace(residuals.min(), residuals.max(), 300)
pdf_y = stats.norm.pdf(xr, residuals.mean(), residuals.std())
fa.add_trace(go.Histogram(
    x=residuals, nbinsx=30, histnorm='probability density',
    marker=dict(color='seagreen',
                line=dict(color='white', width=0.5)),
    opacity=0.8, name='Histogram',
    hovertemplate='Resid=%{x:.0f}<br>Density=%{y:.5f}<extra></extra>'),
    row=2, col=1)
fa.add_trace(go.Scatter(
    x=xr, y=pdf_y, mode='lines',
    line=dict(color='crimson', width=1.8),
    name='Normal curve'), row=2, col=1)
fa.update_xaxes(title_text='Residuals', row=2, col=1)
fa.update_yaxes(title_text='Density',   row=2, col=1)

# 5d. Actual vs Predicted
mn, mx = y.min(), y.max()
fa.add_trace(go.Scatter(
    x=fitted, y=y, mode='markers',
    marker=dict(color='darkorange', size=5, opacity=0.5),
    hovertemplate='Pred=%{x:.0f}<br>Actual=%{y:.0f}<extra></extra>',
    name='Actual vs Pred'), row=2, col=2)
fa.add_trace(go.Scatter(
    x=[mn, mx], y=[mn, mx], mode='lines',
    line=dict(color='red', dash='dash', width=1.8),
    name='Perfect fit'), row=2, col=2)
fa.update_xaxes(title_text='Predicted (ŷ)', row=2, col=2)
fa.update_yaxes(title_text='Actual (y)',    row=2, col=2)

fa.update_layout(
    template=THEME, height=650, showlegend=False,
    title=dict(
        text=(f'<b>Model Adequacy – Birth Weight  (patsy)</b><br>'
              f'<sup>Formula: {FORMULA}  |  '
              f'R²={model.rsquared:.4f}  '
              f'Adj R²={model.rsquared_adj:.4f}  '
              f'RMSE={np.sqrt(model.mse_resid):.1f} g  '
              f'n={n}</sup>'),
        x=0.5, font=dict(size=13)),
    margin=dict(l=60, r=40, t=90, b=60))
fa.show()

# ══════════════════════════════════════════════════════════════
# 6. MODEL DIAGNOSTICS  (2 × 3, 6th hidden)
# ══════════════════════════════════════════════════════════════
fl   = hat_diag > cut_lev
fc   = cooks_d  > cut_cook
fdf  = np.abs(dffits_val) > cut_dff
fcv  = (covratio_val > cut_cov_u) | (covratio_val < cut_cov_l)
dfbg = dfbetas_val[:, 3]          # gestwks column (index 3)
fdb  = np.abs(dfbg) > cut_dfb

fd = make_subplots(
    rows=2, cols=3,
    subplot_titles=[
        '① Leverage (h_ii)',
        "② Cook's Distance (D_i)",
        '③ |DFFITS|',
        '④ COVRATIO',
        '⑤ DFBETAS – gestwks',
        ''],
    vertical_spacing=0.18, horizontal_spacing=0.10)

# Plot 1 : Leverage
xn, yn, xf, yf = split(obs, hat_diag, fl)
fd.add_trace(stems(obs, hat_diag, 'steelblue'), row=1, col=1)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='steelblue', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>h_ii=%{y:.5f}<extra></extra>'),
    row=1, col=1)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f'Leverage > {cut_lev:.4f}',
    hovertemplate='Obs %{x}<br>h_ii=%{y:.5f} ⚠<extra></extra>'),
    row=1, col=1)
fd.add_hline(y=cut_lev,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'Cutoff={cut_lev:.4f}',
    annotation_font_color=FLAG, row=1, col=1)
fd.update_xaxes(title_text='Obs Index', row=1, col=1)
fd.update_yaxes(title_text='h_ii',      row=1, col=1)

# Plot 2 : Cook's D
xn, yn, xf, yf = split(obs, cooks_d, fc)
fd.add_trace(stems(obs, cooks_d, 'darkorange'), row=1, col=2)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='darkorange', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>D=%{y:.6f}<extra></extra>'),
    row=1, col=2)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f"Cook's D > {cut_cook}",
    hovertemplate='Obs %{x}<br>D=%{y:.6f} ⚠<extra></extra>'),
    row=1, col=2)
fd.add_hline(y=cut_cook,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'Cutoff={cut_cook}',
    annotation_font_color=FLAG, row=1, col=2)
fd.update_xaxes(title_text='Obs Index', row=1, col=2)
fd.update_yaxes(title_text='D_i',       row=1, col=2)

# Plot 3 : DFFITS
abs_dff        = np.abs(dffits_val)
xn, yn, xf, yf = split(obs, abs_dff, fdf)
fd.add_trace(stems(obs, abs_dff, '#7B1FA2'), row=1, col=3)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='#7B1FA2', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>|DFFITS|=%{y:.4f}<extra></extra>'),
    row=1, col=3)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f'|DFFITS| > {cut_dff:.4f}',
    hovertemplate='Obs %{x}<br>|DFFITS|=%{y:.4f} ⚠<extra></extra>'),
    row=1, col=3)
fd.add_hline(y=cut_dff,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'Cutoff={cut_dff:.4f}',
    annotation_font_color=FLAG, row=1, col=3)
fd.update_xaxes(title_text='Obs Index',  row=1, col=3)
fd.update_yaxes(title_text='|DFFITS_i|', row=1, col=3)

# Plot 4 : COVRATIO
xn, yn, xf, yf = split(obs, covratio_val, fcv)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='#00897B', size=5, opacity=0.65),
    showlegend=False,
    hovertemplate='Obs %{x}<br>COVRATIO=%{y:.4f}<extra></extra>'),
    row=2, col=1)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name='COVRATIO flagged',
    hovertemplate='Obs %{x}<br>COVRATIO=%{y:.4f} ⚠<extra></extra>'),
    row=2, col=1)
fd.add_hline(y=cut_cov_u,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'Upper={cut_cov_u:.4f}',
    annotation_font_color=FLAG, row=2, col=1)
fd.add_hline(y=cut_cov_l,
    line=dict(color='royalblue', dash='dash', width=1.5),
    annotation_text=f'Lower={cut_cov_l:.4f}',
    annotation_font_color='royalblue',
    annotation_position='bottom right', row=2, col=1)
fd.add_hline(y=1.0,
    line=dict(color='gray', dash='dot', width=1), row=2, col=1)
fd.update_xaxes(title_text='Obs Index',  row=2, col=1)
fd.update_yaxes(title_text='COVRATIO_i', row=2, col=1)

# Plot 5 : DFBETAS – gestwks
xn, yn, xf, yf = split(obs, dfbg, fdb)
fd.add_trace(stems(obs, dfbg, '#2E7D32'), row=2, col=2)
fd.add_trace(go.Scatter(x=xn, y=yn, mode='markers',
    marker=dict(color='#2E7D32', size=5, opacity=0.7),
    showlegend=False,
    hovertemplate='Obs %{x}<br>DFBETAS=%{y:.4f}<extra></extra>'),
    row=2, col=2)
fd.add_trace(go.Scatter(x=xf, y=yf, mode='markers',
    marker=dict(color=FLAG, size=8, symbol='diamond'),
    name=f'|DFBETAS| > {cut_dfb:.4f}',
    hovertemplate='Obs %{x}<br>DFBETAS=%{y:.4f} ⚠<extra></extra>'),
    row=2, col=2)
fd.add_hline(y= cut_dfb,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'+Cutoff={cut_dfb:.4f}',
    annotation_font_color=FLAG, row=2, col=2)
fd.add_hline(y=-cut_dfb,
    line=dict(color=FLAG, dash='dash', width=1.5),
    annotation_text=f'-Cutoff={-cut_dfb:.4f}',
    annotation_font_color=FLAG,
    annotation_position='bottom right', row=2, col=2)
fd.add_hline(y=0,
    line=dict(color='gray', dash='dot', width=1), row=2, col=2)
fd.update_xaxes(title_text='Obs Index', row=2, col=2)
fd.update_yaxes(title_text='DFBETAS',   row=2, col=2)

# Hide 6th cell
fd.update_xaxes(visible=False, row=2, col=3)
fd.update_yaxes(visible=False, row=2, col=3)

fd.update_layout(
    template=THEME, height=700,
    title=dict(
        text=(f'<b>Regression Diagnostics – Birth Weight  (patsy)</b><br>'
              f'<sup>Formula: {FORMULA}  |  '
              f'n={n}  k={k}  |  '
              f'Leverage>{cut_lev:.4f}  '
              f"Cook's D>{cut_cook}  "
              f'|DFFITS|>{cut_dff:.4f}  '
              f'|DFBETAS|>{cut_dfb:.4f}</sup>'),
        x=0.5, font=dict(size=13)),
    legend=dict(
        orientation='h', y=-0.12,
        xanchor='center', x=0.5,
        font=dict(size=10)),
    margin=dict(l=60, r=40, t=100, b=80))

fd.show()

```
## **SAS** :

```
/*================================================================
  Regression Analysis – Birth Weight Data
================================================================*/

/* ══════════════════════════════════════════════════════════════
    IMPORT DATA
   ══════════════════════════════════════════════════════════════ */
proc import
    datafile = "/home/u64098875/sasuser.v94/BirthWt.csv"
    out      = birthwt_raw
    dbms     = csv
    replace;
    getnames = yes;
run;

proc print data=birthwt_raw (obs=5);
    title "First 5 rows – BirthWt";
run;

/* ══════════════════════════════════════════════════════════════
    ENCODE CATEGORICAL VARIABLES
   ══════════════════════════════════════════════════════════════ */
data birthwt;
    set birthwt_raw;
    if      upcase(strip(sex)) = 'MALE'   then sex_male = 1;
    else if upcase(strip(sex)) = 'FEMALE' then sex_male = 0;
    else sex_male = .;

    if      upcase(strip(ht)) = 'YES' then ht_yes = 1;
    else if upcase(strip(ht)) = 'NO'  then ht_yes = 0;
    else ht_yes = .;

    obs = _n_;
run;

/* ══════════════════════════════════════════════════════════════
    FIT OLS MODEL — PROC REG
   ══════════════════════════════════════════════════════════════ */
proc reg data=birthwt;
    model bweight = matage ht_yes gestwks sex_male;
run;
quit;
```