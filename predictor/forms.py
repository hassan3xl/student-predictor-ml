from django import forms

from .mlkit import ALL_FEATURES
from .models import Student

SECTIONS = [
    ("Student", ["name", "sex", "age", "address", "school"]),
    ("Family", ["famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "guardian", "famrel"]),
    ("School and study", ["reason", "traveltime", "studytime", "failures", "schoolsup",
                          "famsup", "paid", "activities", "nursery", "higher", "internet",
                          "romantic"]),
    ("Lifestyle", ["freetime", "goout", "Dalc", "Walc", "health", "absences"]),
    ("Grades (0-20)", ["G1", "G2", "G3"]),
]


class StudentForm(forms.ModelForm):
    class Meta:
        model = Student
        fields = ["name"] + ALL_FEATURES + ["G3"]

    def sections(self):
        return [(title, [self[name] for name in names]) for title, names in SECTIONS]


class CSVUploadForm(forms.Form):
    file = forms.FileField(label="CSV file", help_text="Comma or semicolon separated, with a header row.")
