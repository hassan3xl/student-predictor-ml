"""CSV import: validate every row with StudentForm, save the good ones, score them."""
import pandas as pd

from .forms import StudentForm
from .mlkit import ALL_FEATURES, CATEGORICAL_FEATURES
from .models import Student
from .services import ModelNotTrained, apply_predictions

REQUIRED = [f for f in ALL_FEATURES if f not in ("G1", "G2")]
LOWERCASE = ["Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid",
             "activities", "nursery", "higher", "internet", "romantic"]


def _clean(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return str(value).strip() if isinstance(value, str) else value


def import_dataframe(df, max_errors=20):
    """Returns {created, skipped, errors, scored}. Bad rows are skipped with a reason."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    for col in LOWERCASE:
        df[col] = df[col].astype("string").str.strip().str.lower()
    for col in set(CATEGORICAL_FEATURES) - set(LOWERCASE):
        df[col] = df[col].astype("string").str.strip()

    students, errors, skipped = [], [], 0
    for i, row in enumerate(df.to_dict("records"), start=2):  # row 1 is the header
        data = {k: _clean(v) for k, v in row.items()}
        form = StudentForm(data={**data, "name": data.get("name") or f"Student {i - 1}"})
        if form.is_valid():
            students.append(form.save(commit=False))
        else:
            skipped += 1
            if len(errors) < max_errors:
                field, msgs = next(iter(form.errors.items()))
                errors.append(f"Row {i}: {field}: {msgs[0]}")

    scored = True
    try:
        apply_predictions(students)
    except ModelNotTrained:
        scored = False
    Student.objects.bulk_create(students)
    return {"created": len(students), "skipped": skipped, "errors": errors, "scored": scored}
