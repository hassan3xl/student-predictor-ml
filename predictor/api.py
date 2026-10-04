"""Django Ninja JSON API. Auth: a logged-in session OR an `X-API-Key` header (set API_KEY env var)."""
import secrets
from typing import Literal, Optional

from django.conf import settings
from django.http import Http404
from ninja import File, NinjaAPI, Schema, UploadedFile
from ninja.security import APIKeyHeader, django_auth
from pydantic import Field

from . import services
from .importer import import_dataframe
from .mlkit import read_table
from .models import Student

YesNo = Literal["yes", "no"]
JobType = Literal["teacher", "health", "services", "at_home", "other"]


class ApiKey(APIKeyHeader):
    param_name = "X-API-Key"

    def authenticate(self, request, key):
        expected = settings.API_KEY
        if key and expected and secrets.compare_digest(key.encode(), expected.encode()):
            return key
        return None


api = NinjaAPI(title="Student Risk API", version="1.0",
               description="Predict which students are at risk of failing.",
               # Order matters: the API key is checked first because session auth enforces CSRF
               # on POST and would reject key-only callers before the key was even looked at.
               auth=[ApiKey(), django_auth])


@api.exception_handler(services.ModelNotTrained)
def model_not_trained(request, exc):
    return api.create_response(request, {"detail": str(exc)}, status=503)


class StudentIn(Schema):
    school: Literal["GP", "MS"]
    sex: Literal["F", "M"]
    age: int = Field(ge=15, le=40)
    address: Literal["U", "R"]
    famsize: Literal["LE3", "GT3"]
    Pstatus: Literal["T", "A"]
    Medu: int = Field(ge=0, le=4)
    Fedu: int = Field(ge=0, le=4)
    Mjob: JobType
    Fjob: JobType
    reason: Literal["home", "reputation", "course", "other"]
    guardian: Literal["mother", "father", "other"]
    traveltime: int = Field(ge=1, le=4)
    studytime: int = Field(ge=1, le=4)
    failures: int = Field(ge=0, le=4)
    schoolsup: YesNo
    famsup: YesNo
    paid: YesNo
    activities: YesNo
    nursery: YesNo
    higher: YesNo
    internet: YesNo
    romantic: YesNo
    famrel: int = Field(ge=1, le=5)
    freetime: int = Field(ge=1, le=5)
    goout: int = Field(ge=1, le=5)
    Dalc: int = Field(ge=1, le=5)
    Walc: int = Field(ge=1, le=5)
    health: int = Field(ge=1, le=5)
    absences: int = Field(ge=0, le=100)
    G1: Optional[int] = Field(None, ge=0, le=20)
    G2: Optional[int] = Field(None, ge=0, le=20)


class PredictionOut(Schema):
    probability: float
    at_risk: bool
    risk_level: str
    model_used: str
    threshold: float


class StudentOut(Schema):
    id: int
    name: str
    age: int
    sex: str
    absences: int
    failures: int
    G1: Optional[int] = None
    G2: Optional[int] = None
    risk_probability: Optional[float] = None
    at_risk: Optional[bool] = None
    risk_level: str
    model_used: str


class UploadOut(Schema):
    created: int
    skipped: int
    scored: bool
    errors: list[str]


@api.post("/predict", response=PredictionOut, summary="Score one student (nothing is saved)")
def predict(request, payload: StudentIn):
    return services.predict_one(payload.dict())


@api.get("/students", response=list[StudentOut], summary="List students")
def list_students(request, at_risk: Optional[bool] = None, limit: int = 100, offset: int = 0):
    qs = Student.objects.all()
    if at_risk is not None:
        qs = qs.filter(at_risk=at_risk)
    limit = max(1, min(limit, 500))
    return list(qs.order_by("-risk_probability", "id")[offset:offset + limit])


@api.get("/students/at-risk", response=list[StudentOut],
         summary="Students flagged at risk, highest probability first")
def at_risk_students(request, min_probability: float = 0.0, limit: int = 100):
    qs = Student.objects.filter(at_risk=True, risk_probability__gte=min_probability)
    return list(qs.order_by("-risk_probability", "id")[:max(1, min(limit, 500))])


@api.get("/students/{student_id}", response=StudentOut, summary="One student")
def get_student(request, student_id: int):
    try:
        return Student.objects.get(pk=student_id)
    except Student.DoesNotExist:
        raise Http404("Student not found")


@api.post("/upload", response=UploadOut, summary="Import a CSV of students and score them")
def upload(request, file: UploadedFile = File(...)):
    try:
        return import_dataframe(read_table(file))
    except ValueError as exc:
        return api.create_response(request, {"detail": str(exc)}, status=422)


@api.get("/metrics", summary="Training metrics of the deployed models")
def metrics(request):
    data = services.get_metrics()
    if data is None:
        raise services.ModelNotTrained("No trained model found. Run: python manage.py train_model")
    return data
