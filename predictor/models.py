from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models

YES_NO = [("yes", "Yes"), ("no", "No")]
MJOB = [("teacher", "Teacher"), ("health", "Health care"), ("services", "Civil services"),
        ("at_home", "At home"), ("other", "Other")]
REASON = [("home", "Close to home"), ("reputation", "School reputation"),
          ("course", "Course preference"), ("other", "Other")]
GUARDIAN = [("mother", "Mother"), ("father", "Father"), ("other", "Other")]


def scale(label, lo, hi, help_text="", **kw):
    return models.IntegerField(
        label, validators=[MinValueValidator(lo), MaxValueValidator(hi)],
        help_text=help_text or f"{lo} to {hi}", **kw)


def yes_no(label):
    return models.CharField(label, max_length=3, choices=YES_NO)


class Student(models.Model):
    """One student record. Column names match the UCI Student Performance dataset so the
    same fields feed the ML pipeline (see predictor/mlkit.py)."""

    name = models.CharField(max_length=120, blank=True)

    # --- Demographics / family ---
    school = models.CharField("School", max_length=2, choices=[("GP", "GP"), ("MS", "MS")])
    sex = models.CharField("Sex", max_length=1, choices=[("F", "Female"), ("M", "Male")])
    age = scale("Age", 15, 40)
    address = models.CharField("Home address type", max_length=1,
                               choices=[("U", "Urban"), ("R", "Rural")])
    famsize = models.CharField("Family size", max_length=3,
                               choices=[("LE3", "3 or fewer"), ("GT3", "More than 3")])
    Pstatus = models.CharField("Parents' status", max_length=1,
                               choices=[("T", "Living together"), ("A", "Apart")])
    Medu = scale("Mother's education", 0, 4, "0 none, 1 primary, 2 lower secondary, 3 secondary, 4 higher")
    Fedu = scale("Father's education", 0, 4, "0 none, 1 primary, 2 lower secondary, 3 secondary, 4 higher")
    Mjob = models.CharField("Mother's job", max_length=10, choices=MJOB)
    Fjob = models.CharField("Father's job", max_length=10, choices=MJOB)
    guardian = models.CharField("Guardian", max_length=6, choices=GUARDIAN)
    famrel = scale("Family relationship quality", 1, 5, "1 very bad to 5 excellent")

    # --- Study habits / school ---
    reason = models.CharField("Reason for choosing school", max_length=10, choices=REASON)
    traveltime = scale("Travel time to school", 1, 4, "1 under 15 min, 2 15-30, 3 30-60, 4 over 60")
    studytime = scale("Weekly study time", 1, 4, "1 under 2h, 2 2-5h, 3 5-10h, 4 over 10h")
    failures = scale("Past class failures", 0, 4, "Number of past failures (4 means 4 or more)")
    schoolsup = yes_no("Extra school support")
    famsup = yes_no("Family educational support")
    paid = yes_no("Paid extra classes")
    activities = yes_no("Extra-curricular activities")
    nursery = yes_no("Attended nursery school")
    higher = yes_no("Wants higher education")
    internet = yes_no("Internet at home")
    romantic = yes_no("In a relationship")

    # --- Lifestyle ---
    freetime = scale("Free time after school", 1, 5, "1 very low to 5 very high")
    goout = scale("Going out with friends", 1, 5, "1 very low to 5 very high")
    Dalc = scale("Workday alcohol use", 1, 5, "1 very low to 5 very high")
    Walc = scale("Weekend alcohol use", 1, 5, "1 very low to 5 very high")
    health = scale("Current health status", 1, 5, "1 very bad to 5 very good")
    absences = scale("School absences", 0, 100, "Number of absences")

    # --- Grades (0-20). G1/G2 are optional; G3 is the final grade once known. ---
    G1 = scale("First period grade (G1)", 0, 20, "Optional. Enables the more accurate model.",
               null=True, blank=True)
    G2 = scale("Second period grade (G2)", 0, 20, "Optional. Enables the more accurate model.",
               null=True, blank=True)
    G3 = scale("Final grade (G3)", 0, 20, "Optional. Fill in once the outcome is known.",
               null=True, blank=True)

    # --- Prediction output (written by predictor.services) ---
    risk_probability = models.FloatField(null=True, blank=True)
    at_risk = models.BooleanField(null=True, blank=True)
    risk_level = models.CharField(max_length=6, blank=True)  # low / medium / high
    model_used = models.CharField(max_length=5, blank=True)  # early / full
    predicted_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return self.name or f"Student #{self.pk}"

    @property
    def display_name(self):
        return self.name or f"Student #{self.pk}"
