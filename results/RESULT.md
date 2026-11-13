# Forecasting the water level at the Point: model results

This folder holds the results for two models, **GBM A** and **GBM AE**, compared with the **persistence
baseline**. It also describes how they differ from the models in the teammate notebook
`src/systems_final_project/model.ipynb`, which is commit `72eca72` on GitHub.

**Bottom line**

- Both models clearly beat persistence at 6, 12 and 24 hours ahead.
- They warn of **31 to 36 of the 40 floods before they start**. Persistence can warn of none, by construction.
- **GBM AE is slightly more accurate. GBM A is the model to deploy**, because the Elizabeth gauge that AE
  needs has been offline since June 2026.

---

## 1. What is being predicted

- **Target:** the water level at **USGS 03085152, the Monongahela at Point State Park**, 6, 12 and 24 hours
  ahead, issued every hour.
- **Alarm level:** **18 ft**, the NWS *action stage* for Pittsburgh. Water starts reaching the Mon Wharf at
  18 ft and the North Shore riverwalk at 19 ft. Flood stage is 25 ft.
- **Flood events:** a flood event is a run of hours at or above 18 ft. Runs less than 24 h apart count as one
  event. There are **40 events** in water years 2016–2027.

## 2. The models

All three predict the **change** in level over the next *h* hours. The forecast is the current level plus that
change. Each horizon has its own model.

| Model | What it is | Inputs |
|---|---|---|
| **Persistence (baseline)** | "The level in *h* hours = the level now." No training. | Point's current level |
| **GBM A** — *the deployable model* | Gradient boosting (scikit-learn `HistGradientBoostingRegressor`) | 52 features, all still reported in real time |
| **GBM AE** — *A + Elizabeth* | The same model and settings as A, plus the Elizabeth gauge | 55 features: A's 52 plus 3 Elizabeth features |

**What "A" and "AE" mean.** **A** is our main input set: the Point's own history, the upstream gauges, observed
rain, season and flags. **AE** is **A plus Elizabeth**, a Monongahela gauge upstream of Braddock that the cleaning
notebook had dropped. AE exists to test whether Elizabeth adds information. It does, slightly. But Elizabeth
stopped reporting in June 2026, so AE cannot run reliably today.

### The 52 features of GBM A

Features are built in `src/systems_final_project/features.py`; the models are in `src/systems_final_project/models.py`.

| Group | Features | Count |
|---|---|---|
| Point's own history | level now, 6 h ago, 24 h ago; rise over the last 1 h and 6 h; highest level in the last 24 h | 6 |
| Upstream gauges | **Natrona, Kittanning** (Allegheny) and **Sutersville** (Youghiogheny): level now, 6 h ago, 12 h ago, and rise over 1 h and 6 h. **Braddock** (Monongahela): rise over 1, 6 and 12 h only, because Dam 3's removal in July 2024 shifted its level | 18 |
| Observed rain (ERA5 via Open-Meteo) | totals over the past 6, 24, 48 and 72 h at Pittsburgh, Kittanning, Elizabeth, Sutersville and their average | 20 |
| Season | day of year as sin and cos | 2 |
| Flags | after Dam 3 removal (`post_dam3`); hour filled from a short gap, one flag per gauge | 6 |

**GBM AE adds** the Elizabeth rise over 1, 6 and 12 h. Like Braddock, it uses rises only because of the Dam 3 shift.

**Settings, shared by A and AE:**

- `learning_rate=0.05`, `max_iter=400`, `max_leaf_nodes=31`, `min_samples_leaf=40`, `l2_regularization=1.0`,
  `random_state=0`.
- `early_stopping=False`. With early stopping on, scikit-learn would hold out a random 10% of hours for
  validation, and neighbouring hours would end up on both sides of that split.
- Missing inputs are left as NaN; the trees handle them directly.

## 3. How the models are evaluated

- **Leave-one-water-year-out:** this is the 12-fold version of the proposal's leave-one-year-out plan.
  - Water years run October to September.
  - Each year from 2016 to 2027 is predicted by a model trained on all the other years.
  - Training rows whose *target* falls in the test year are dropped.
  - Every scored hour therefore comes from a model that never saw its year.
- **Scored hours:** all hours where the current and future levels are known, about **97,200 per horizon**. Every
  model is scored on exactly the same hours.
- **Metrics:**

  | Metric | What it measures |
  |---|---|
  | RMSE | error over all hours |
  | High-water MAE | error when observed ≥ 18 ft |
  | Rising MAE | error when the river rises ≥ 1 ft within the horizon |
  | Precision, recall, CSI | hourly alarm skill at 18 ft. CSI = hits / (hits + misses + false alarms) |
  | Events warned early | out of 40: an alarm issued **before** the event starts, for a time inside the event |
  | Lead time | how many hours before the event start that first alarm was issued |

## 4. Results (leave-one-water-year-out, 2016–2027, 40 events)

![Model accuracy](figures/fig1_accuracy.png)

| Horizon | Model | RMSE (ft) | High-water MAE (ft) | Rising MAE (ft) | Precision | Recall | CSI | Events warned early | Median lead |
|---|---|---|---|---|---|---|---|---|---|
| 6 h | Persistence | 0.260 | 0.792 | 1.614 | 0.862 | 0.862 | 0.758 | 0 / 40 | 0 h |
| 6 h | **GBM A** | 0.161 | 0.290 | 0.631 | 0.959 | 0.902 | 0.868 | **36 / 40** | 3 h |
| 6 h | GBM AE | **0.154** | **0.265** | **0.601** | 0.955 | 0.906 | **0.869** | 35 / 40 | 3 h |
| 12 h | Persistence | 0.419 | 1.495 | 2.092 | 0.740 | 0.740 | 0.587 | 0 / 40 | 0 h |
| 12 h | **GBM A** | 0.278 | 0.722 | 1.112 | 0.891 | 0.819 | 0.744 | 34 / 40 | 7 h |
| 12 h | GBM AE | **0.268** | **0.682** | **1.072** | 0.890 | 0.829 | **0.752** | 34 / 40 | 7 h |
| 24 h | Persistence | 0.597 | 2.473 | 2.667 | 0.555 | 0.558 | 0.386 | 0 / 40 | 0 h |
| 24 h | **GBM A** | 0.440 | 1.634 | 1.936 | 0.676 | 0.621 | 0.479 | **31 / 40** | 12 h |
| 24 h | GBM AE | **0.436** | **1.604** | **1.925** | 0.684 | 0.631 | **0.489** | 30 / 40 | 12 h |

**Reading the table**

- **GBM A vs persistence:**
  - RMSE is 38% lower at 6 h, 34% at 12 h and 26% at 24 h.
  - High-water error is 63%, 52% and 34% lower.
- **24 h is the weakest horizon**, as the proposal expected. In April 2024 the 24 h forecast under-predicts
  both peaks and is late (see the figure below).

### GBM A vs GBM AE: which is better?

- **Accuracy:** AE has slightly lower error at every horizon.
  - RMSE is 0.004–0.010 ft lower.
  - High-water MAE is 0.025–0.040 ft lower.
  - The gain is smallest at 24 h.
- **Alarms:** they are effectively equal.
  - "Events warned early" differs by at most one event: 36 vs 35 at 6 h, 31 vs 30 at 24 h.
  - Hourly CSI is the same or slightly higher for AE.
- **Deployability:** only A can run today.
  - Elizabeth has been offline since 2026-06-07.
  - Since then the Point has peaked at only 18.15 ft, so AE's behaviour in a real flood without Elizabeth is
    untested.
- **Recommendation:**
  - Use **GBM A** for the forecast, alarms and dashboard.
  - Report **GBM AE** as evidence that Elizabeth adds a small benefit.
  - Switch to AE if Elizabeth comes back online.

### April 2024: the highest level in the record (28.3 ft), in a held-out year

![April 2024 forecasts](figures/fig2_april_2024.png)

- **6 h:** both models track the rise and the peak almost exactly. Persistence is the observed curve shifted
  6 h late.
- **12 h:** the models are slightly low and late on the rise, but far closer than persistence.
- **24 h:** the models rise before persistence but under-predict both peaks by 3–4 ft.

## 5. Comparison with the teammate model (`model.ipynb`)

### How the two approaches differ

| | `model.ipynb` | Ours (GBM A / AE) |
|---|---|---|
| Models | persistence, persistence + 6 h trend, `LinearRegression`, `HistGradientBoostingRegressor` | persistence, GBM A, GBM AE |
| GBM settings | `max_iter=300`, `learning_rate=0.05`, `max_leaf_nodes=31`, otherwise defaults. Early stopping is on by default, so a random 10% of hours is held out | `max_iter=400`, `min_samples_leaf=40`, `l2_regularization=1.0`, early stopping off |
| Target | 6 h change in level (only 6 h was run) | change in level at 6, 12 and 24 h |
| Features | **25** | **52** for A, **55** for AE |
| Upstream gauges | changes only (1/3/6 h) at Braddock, Sutersville, Natrona, Kittanning | changes, **plus absolute levels (now, 6 h, 12 h ago)** at Natrona, Kittanning, Sutersville. Braddock uses changes only |
| Point history | level, 1/3/6 h changes | level, 6 h and 24 h ago, 1/6 h changes, **24 h max** |
| Rain | 6/24/48 h totals: Allegheny (= Kittanning), Mon (average of Elizabeth and Sutersville), Pittsburgh | 6/24/48/**72 h** totals at 4 sites separately plus their average |
| Season and flags | none | day of year, Dam 3 flag, gap-fill flags |
| Elizabeth | left out | left out in A, used in AE |
| Validation | walk-forward: each year from 2019 to 2026 is trained only on earlier years | leave-one-water-year-out over all of 2016–2027 |
| Alarm level | **20 ft** | **18 ft** (NWS action stage) |
| Alarm scoring | event "caught" if the alarm is on in the 24 h before onset; counts false-alarm *periods* | event warned if an alarm issued before onset targets a time inside the event; hourly precision, recall and CSI |

The two share **17 identical features**:

- Point's level and 1 h and 6 h changes;
- the 1 h and 6 h changes at the four upstream gauges;
- the 6/24/48 h rain totals at Kittanning and Pittsburgh.

The notebook has 8 features we don't: 3 h changes and the averaged Mon rain. We have 35 it doesn't.

The notebook itself is on GitHub but not in the current local checkout, which is at `b2bce37`.

### To agree on as a team

1. **Alarm level:** 18 ft (NWS action stage) or 20 ft.
2. **Event scoring rule:** one rule for the report.
3. **Main validation scheme:** leave-one-year-out, as in the proposal, or walk-forward. Leave-one-year-out also
   trains on years after the test year, so it may be somewhat optimistic.

## 6. Caveats

- **Rain data:** observed rain is ERA5, which is published about 5 days late. These results assume rain is known
  up to the forecast time. A real-time system needs a live rain source.
- **Few events:** only 40 floods in 11 years, so differences of one or two events are noise.

## 7. Reproduce

From the repository root, after `uv sync`:

```bash
uv run python -m systems_final_project.features      # 1. data/features/
uv run python -m systems_final_project.models --force  # 2. results/predictions/ (~8 min)
uv run python -m systems_final_project.evaluation    # 3. results/figures/
```

Without `--force`, step 2 keeps any model that already has a predictions file.

### Code layout (`src/systems_final_project/`)

```
data/config.py    every path, gauge ID and date range (shared with the teammate download code)
data/             downloads and cleaning (teammate code; produces data/clean/hourly.parquet)
features.py       model inputs and targets              -> data/features/
models.py         gbm_A, gbm_AE, persistence            -> results/predictions/
evaluation.py     scoring + the two figures             -> results/figures/
```

### Outputs (`results/`)

| Path | Contents |
|---|---|
| `predictions/{gbm_A,gbm_AE,persistence}.parquet` | hourly held-out forecasts for each model (`pred_6h`, `pred_12h`, `pred_24h`) |
| `figures/fig1_accuracy.png` | accuracy over all 40 flood events (section 4) |
| `figures/fig2_april_2024.png` | forecasts during the April 2024 floods (section 4) |
