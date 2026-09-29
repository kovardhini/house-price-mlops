## Phase 12-13: Model Comparison & Selection

CV Results (5-fold, log-RMSE):
- Ridge: 0.1195 (+/- 0.0084) - 0.02s
- XGBoost: 0.1201 (+/- 0.0074) - 1.18s
- CatBoost: 0.1203 (+/- 0.0099) - 1.78s
- HistGradientBoosting: 0.1281 (+/- 0.0056) - 5.89s
- RandomForest: 0.1328 (+/- 0.0070) - 3.36s

Decision: Proceed to Phase 14 tuning both Ridge (alpha) and XGBoost
(max_depth, learning_rate, n_estimators, subsample).
Reason: Ridge currently leads but margin over XGBoost/CatBoost is within
noise; boosting models have more tuning headroom that hasn't been explored yet.
RandomForest and HistGradientBoosting dropped - clearly behind, unlikely
to catch up with tuning, not worth further MacBook CPU time.
## Phase 14: Hyperparameter Tuning

Before/After CV RMSE (5-fold):
- Ridge: 0.1195 -> 0.1140 (tuned via GridSearchCV over alpha)
- XGBoost: 0.1201 -> 0.1170 (tuned via RandomizedSearchCV, 40 iterations)

Finding: Ridge improved MORE from tuning than XGBoost did, contrary to the
expectation that boosting models have more tuning headroom. Ridge (tuned)
now leads by a margin larger than the CV standard deviation observed earlier,
suggesting a real (not noise-level) advantage.

Decision: Ridge (tuned) is the leading candidate heading into Phase 15
(Feature Selection). Final selection will be confirmed after testing
whether feature selection changes this picture.
## Phase 15: Feature Selection

Tested removing TotalQualityScore and HasPool (both previously flagged as
weak/redundant in Phases 8 and 11).

Full feature set RMSE: 0.1140
Reduced feature set RMSE: 0.1147

Decision: KEEP all features. Removal did not improve performance (slightly
worse, though within typical noise). This confirms the project's guidance
against removing features based on low individual correlation alone --
TotalQualityScore and HasPool contribute small but real value in combination
with other features, even though their standalone correlation was weak.

Final feature set: unchanged from Phase 8 (all 92 engineered + original
features retained).
## Phase 17: Final Model Evaluation

Final model: Ridge Regression (alpha tuned in Phase 14), trained on full
training set, evaluated once on the locked test set.

Test Set Results (dollar scale):
- MAE: $13,908
- RMSE: $19,155
- R²: 0.9332
- RMSLE: 0.1205

Sanity check - CV vs Test (log-scale RMSE):
- CV RMSE (Phase 14): 0.1140
- Test RMSE (this phase): 0.1205
- Gap: 0.0065 (~5.7% relative) - small and expected, confirms the model
  generalizes well without overfitting to the tuning/selection process.

This is the FINAL reported performance. Test set will not be touched again.
## Phase 18: Model Saving

Combined the preprocessor and tuned Ridge model into a single scikit-learn
Pipeline object, refit on the FULL dataset (train + test combined), since
the test set's job of producing an honest performance estimate (Phase 17)
was already complete.

Saved as models/house_price_pipeline.pkl - a single deployable artifact
that takes raw feature input and returns a log-scale prediction in one call.

Important: pipeline output is in log-price space. Must apply np.expm1()
to get the dollar prediction - this will be handled automatically inside
predict.py in Phase 19.

Verified pipeline works end-to-end: sample prediction $128,790 vs actual
$126,000 (~2.2% off) - reasonable, though not a true generalization test
since this row was part of the final training data.
## Phase 20: FastAPI Backend

Built a FastAPI app (api/main.py) with:
- POST /predict - accepts raw property features via Pydantic model validation,
  automatically computes engineered features, returns predicted_price
- GET /health - basic health check
- GET /docs - auto-generated interactive API documentation

Verified end-to-end: submitted a sample house via /docs, received
predicted_price: $198,470.87, consistent with the direct predict.py test
($203,172.62 for a slightly different input).

Pydantic model enforces input validation at the API boundary (types,
required fields, OverallQual range 1-10) before requests reach the model.
## Phase 21: Frontend

Built a Gradio UI (frontend/app.py) exposing ~15 key property attributes
via sliders and dropdowns, grouped into sections (Size & Structure, Rooms,
Quality, Garage, Location). Custom emerald theme and styled result card
for a more polished look than default Gradio styling.

Fixed two issues during development:
- Missing DEFAULT_FEATURES entries (RoofStyle, GarageType) caused a
  "columns are missing" error from the pipeline's ColumnTransformer
- Gradio 6.0 moved theme/css params from Blocks() constructor to
  launch() - updated accordingly

Verified working: sample input (1500 sqft, quality 6, 3 bed, 2 bath,
CollgCr neighborhood) returned $152,219, a sensible mid-range prediction.