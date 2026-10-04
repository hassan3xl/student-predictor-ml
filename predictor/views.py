from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from . import services
from .forms import CSVUploadForm, StudentForm
from .importer import import_dataframe
from .mlkit import ALL_FEATURES, read_table
from .models import Student

NOT_TRAINED = "No trained model yet. Run `python manage.py train_model`, then re-score from the dashboard."


@login_required
def dashboard(request):
    qs = Student.objects.all()
    total = qs.count()
    scored = qs.filter(risk_probability__isnull=False)
    at_risk = qs.filter(at_risk=True).count()

    levels = {row["risk_level"]: row["n"] for row in
              scored.values("risk_level").annotate(n=Count("id"))}
    bands = [0] * 5
    for p in scored.values_list("risk_probability", flat=True):
        bands[min(int(p * 5), 4)] += 1

    context = {
        "total": total,
        "scored": scored.count(),
        "at_risk": at_risk,
        "at_risk_pct": round(100 * at_risk / total, 1) if total else 0,
        "top": qs.filter(at_risk=True).order_by("-risk_probability")[:10],
        "metrics": services.get_metrics(),
        "chart": {
            "levels": [levels.get("low", 0), levels.get("medium", 0), levels.get("high", 0)],
            "bands": bands,
        },
    }
    return render(request, "predictor/dashboard.html", context)


@login_required
def student_list(request):
    qs = Student.objects.all()
    risk = request.GET.get("risk", "")
    q = request.GET.get("q", "").strip()
    if risk == "flagged":
        qs = qs.filter(at_risk=True)
    elif risk in ("low", "medium", "high"):
        qs = qs.filter(risk_level=risk)
    elif risk == "unscored":
        qs = qs.filter(risk_probability__isnull=True)
    if q:
        match = Q(name__icontains=q)
        if q.isdigit():
            match |= Q(pk=int(q))
        qs = qs.filter(match)
    qs = qs.order_by("-risk_probability", "id")
    page = Paginator(qs, 25).get_page(request.GET.get("page"))
    return render(request, "predictor/student_list.html", {"page": page, "risk": risk, "q": q})


@login_required
def student_add(request):
    form = StudentForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        student = form.save(commit=False)
        try:
            services.apply_predictions([student])
        except services.ModelNotTrained:
            messages.warning(request, NOT_TRAINED)
        student.save()
        messages.success(request, "Student saved.")
        return redirect("student_detail", pk=student.pk)
    return render(request, "predictor/student_form.html", {"form": form})


@login_required
def student_detail(request, pk):
    student = get_object_or_404(Student, pk=pk)
    metrics = services.get_metrics()
    threshold = metrics["variants"][student.model_used]["threshold"] if metrics and student.model_used else None
    shown = set(ALL_FEATURES) | {"G3"}
    details = [
        (f.verbose_name, getattr(student, f"get_{f.name}_display")() if f.choices
         else getattr(student, f.name))
        for f in Student._meta.fields if f.name in shown
    ]
    return render(request, "predictor/student_detail.html",
                  {"s": student, "threshold": threshold, "details": details})


@login_required
@require_POST
def student_predict(request, pk):
    student = get_object_or_404(Student, pk=pk)
    try:
        services.apply_predictions([student])
        student.save()
        messages.success(request, "Prediction refreshed.")
    except services.ModelNotTrained:
        messages.error(request, NOT_TRAINED)
    return redirect("student_detail", pk=pk)


@login_required
@require_POST
def predict_all(request):
    students = list(Student.objects.all())
    try:
        services.apply_predictions(students)
    except services.ModelNotTrained:
        messages.error(request, NOT_TRAINED)
        return redirect("dashboard")
    Student.objects.bulk_update(
        students, ["risk_probability", "at_risk", "risk_level", "model_used", "predicted_at"])
    messages.success(request, f"Re-scored {len(students)} students.")
    return redirect("dashboard")


@login_required
def upload(request):
    form = CSVUploadForm()
    result = None
    if request.method == "POST":
        try:
            if "sample" in request.POST:
                if not settings.SAMPLE_DATA.exists():
                    raise ValueError("Sample dataset file not found.")
                result = import_dataframe(read_table(settings.SAMPLE_DATA))
            else:
                form = CSVUploadForm(request.POST, request.FILES)
                if form.is_valid():
                    result = import_dataframe(read_table(request.FILES["file"]))
        except ValueError as exc:
            messages.error(request, str(exc))
        if result:
            if not result["scored"]:
                messages.warning(request, NOT_TRAINED)
            messages.success(request, f"Imported {result['created']} students "
                                      f"({result['skipped']} rows skipped).")
    return render(request, "predictor/upload.html",
                  {"form": form, "result": result, "has_sample": settings.SAMPLE_DATA.exists()})


@login_required
def metrics_page(request):
    return render(request, "predictor/metrics.html", {"m": services.get_metrics()})
