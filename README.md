# Predictive Analysis of Student Academic Performance (Django + Django Ninja + scikit-learn)

Flags students at risk of a final grade below 10/20 so staff can intervene early.
Server-rendered Django templates for staff, a Django Ninja JSON API for other systems,
and scikit-learn models trained offline.

## Quick start

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

python manage.py train_model          # ~30 s. Retrain on YOUR machine (see note below)
python manage.py migrate
python manage.py createsuperuser      # staff log in through the Django admin login page
python manage.py import_students      # loads the bundled UCI sample (395 students) and scores them
python manage.py runserver
```

Open http://127.0.0.1:8000/ and log in. API docs are at http://127.0.0.1:8000/api/docs.

> Always run `train_model` after installing. Saved models are tied to the scikit-learn version
> that created them; if yours differs the app tells you to retrain rather than crashing.

## Pages

| URL | What it does |
|---|---|
| `/` | Dashboard: counts, risk-level chart, probability histogram, top 10 at-risk students |
| `/students/` | Searchable, filterable list (flagged / high / medium / low / not scored) |
| `/students/add/` | Form that validates, saves and predicts one student |
| `/students/<id>/` | Prediction, which model scored it, full record, re-run button |
| `/upload/` | CSV import with per-row validation and error report |
| `/metrics/` | Data-quality summary, cross-validation table, hold-out results, confusion matrix, feature importance |
| `/admin/` | Edit records, bulk "re-run prediction" action |

## API (Django Ninja)

Auth: a logged-in session, **or** an `X-API-Key` header matching the `API_KEY` environment variable
(`API_KEY=some-secret python manage.py runserver`). With no `API_KEY` set, key access is disabled.

| Endpoint | Purpose |
|---|---|
| `POST /api/predict` | Score one student from JSON (not saved) |
| `GET /api/students?at_risk=true&limit=&offset=` | List students |
| `GET /api/students/at-risk?min_probability=0.6` | Flagged students, highest risk first |
| `GET /api/students/{id}` | One student |
| `POST /api/upload` | Multipart CSV import, returns created/skipped counts and row errors |
| `GET /api/metrics` | Training metrics |

```bash
curl -H "X-API-Key: some-secret" "http://127.0.0.1:8000/api/students/at-risk?limit=5"
```

## How it maps to the objectives

1. **Collect, store, manage data:** `Student` model with field-level validation, Django admin, CSV import (web, API and command line).
2. **Cleanse and preprocess:** one scikit-learn `Pipeline` per model: median/most-frequent imputation, IQR outlier clipping, scaling, one-hot encoding. Everything is fitted inside cross-validation folds, so there is no data leakage. `/metrics/` reports missing cells, duplicates and outliers found.
3. **Identify at-risk students:** Logistic Regression, Random Forest and Gradient Boosting are compared with repeated stratified 5-fold CV on a training split; the best by ROC-AUC is evaluated on a held-out 20% test set. The alert threshold is the highest cut-off that still catches at least 80% of at-risk students (out-of-fold), because missing a struggling student costs more than a false alarm.
4. **Integrate into workflows:** the same service layer (`predictor/services.py`) powers the staff UI and the API; other systems call the API with a key and bulk-load via CSV.

## Two models, on purpose

| Model | Uses | Use it when |
|---|---|---|
| Early-warning | background, habits, past failures, absences | before any assessment exists |
| Grades | all of the above plus first/second period grades (G1, G2) | after at least two assessments |

A student is scored with the grades model automatically when both G1 and G2 are present. Period grades
dominate that model, so its accuracy is high but it is less useful for *early* intervention. The
early-warning model is weaker. Check `/metrics/` for the exact numbers from your training run, and
report both in your write-up.

## Using your department's real data

The bundled data is the public UCI Student Performance dataset (Portuguese secondary schools, maths course),
a stand-in until you have local records. To retrain on your own data:

1. Make a CSV with the same column names (see `/upload/` for the list) plus `G3` (final grade, 0-20).
2. `python manage.py train_model --data path/to/your.csv`
3. `python manage.py import_students --file path/to/your.csv --clear`

Columns such as `school`, `reason` and `guardian` are specific to the UCI survey. If your records use
different variables, edit the feature lists in `predictor/mlkit.py`, the matching fields in
`predictor/models.py`, and `StudentIn` in `predictor/api.py`, then rerun the steps above.

## Limits to state honestly

- With only ~400 rows, the hold-out test set is about 80 students, so single metrics move by several points. Prefer the cross-validation numbers.
- Predictions are probabilities from historical patterns, not verdicts. Use them to prioritise conversations with students.
- Cut-offs, thresholds and "at risk = final grade below 10/20" are choices; change `PASS_MARK` in `predictor/mlkit.py` to suit your grading scale.

## Tests and deployment notes

```bash
python manage.py test predictor        # 12 tests: import, validation, API auth, pages, scoring
```

For production set `DJANGO_DEBUG=0`, `DJANGO_SECRET_KEY`, `DJANGO_ALLOWED_HOSTS`, run `collectstatic`,
and serve with gunicorn behind a web server.

## Layout

```
config/                settings, urls
predictor/mlkit.py     features, IQR clipper, pipeline builder (no Django imports)
predictor/training.py  offline training, evaluation, plots
predictor/services.py  loads models, scores students (used by views and API)
predictor/api.py       Django Ninja endpoints
predictor/views.py     server-rendered pages        predictor/templates/
predictor/importer.py  validated CSV import
ml_artifacts/          saved models + metrics.json
data/student-mat.csv   bundled UCI sample
```
