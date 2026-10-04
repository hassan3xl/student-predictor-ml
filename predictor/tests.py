import json
from unittest import skipUnless

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from . import services
from .forms import StudentForm
from .importer import import_dataframe
from .mlkit import read_table
from .models import Student

TRAINED = (settings.ML_ARTIFACT_DIR / "model_early.joblib").exists()

GOOD = {
    "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3", "Pstatus": "T",
    "Medu": 3, "Fedu": 2, "Mjob": "other", "Fjob": "services", "reason": "course",
    "guardian": "mother", "traveltime": 1, "studytime": 2, "failures": 0, "schoolsup": "no",
    "famsup": "yes", "paid": "no", "activities": "yes", "nursery": "yes", "higher": "yes",
    "internet": "yes", "romantic": "no", "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1,
    "Walc": 2, "health": 4, "absences": 4,
}


class BaseCase(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_superuser("staff", "s@example.com", "pw12345!")
        self.client.force_login(self.user)


class ImportTests(TestCase):
    @skipUnless(TRAINED and settings.SAMPLE_DATA.exists(), "needs trained model and sample data")
    def test_sample_import_creates_and_scores_all_rows(self):
        result = import_dataframe(read_table(settings.SAMPLE_DATA))
        self.assertEqual(result["created"], 395)
        self.assertEqual(result["skipped"], 0)
        self.assertEqual(Student.objects.filter(risk_probability__isnull=True).count(), 0)
        self.assertGreater(Student.objects.filter(at_risk=True).count(), 0)

    def test_bad_rows_are_skipped_with_a_reason(self):
        import pandas as pd
        df = pd.DataFrame([GOOD, {**GOOD, "sex": "X"}, {**GOOD, "age": 3}])
        result = import_dataframe(df)
        self.assertEqual((result["created"], result["skipped"]), (1, 2))
        self.assertEqual(len(result["errors"]), 2)

    def test_missing_required_column_raises(self):
        import pandas as pd
        with self.assertRaises(ValueError):
            import_dataframe(pd.DataFrame([{"age": 17}]))


class FormTests(TestCase):
    def test_form_accepts_valid_and_rejects_out_of_range(self):
        self.assertTrue(StudentForm(data={**GOOD, "name": "A"}).is_valid())
        self.assertFalse(StudentForm(data={**GOOD, "famrel": 9}).is_valid())


@skipUnless(TRAINED, "model not trained")
class ServiceTests(TestCase):
    def test_grades_route_to_full_model_and_good_grades_are_safer(self):
        early = services.predict_one(GOOD)
        low = services.predict_one({**GOOD, "G1": 5, "G2": 4})
        high = services.predict_one({**GOOD, "G1": 15, "G2": 16})
        self.assertEqual(early["model_used"], "early")
        self.assertEqual(low["model_used"], "full")
        self.assertTrue(low["at_risk"])
        self.assertFalse(high["at_risk"])
        self.assertGreater(low["probability"], high["probability"])


@skipUnless(TRAINED, "model not trained")
@override_settings(API_KEY="test-key")
class ApiTests(BaseCase):
    def post(self, payload, **kw):
        return self.client.post("/api/predict", json.dumps(payload),
                                content_type="application/json", **kw)

    def test_predict_ok(self):
        resp = self.post({**GOOD, "G1": 6, "G2": 5})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["at_risk"])

    def test_predict_validation_error(self):
        self.assertEqual(self.post({**GOOD, "age": 3}).status_code, 422)

    def test_anonymous_is_rejected_but_api_key_works(self):
        self.client.logout()
        self.assertEqual(self.post(GOOD).status_code, 401)
        self.assertEqual(self.post(GOOD, HTTP_X_API_KEY="test-key").status_code, 200)
        self.assertEqual(self.post(GOOD, HTTP_X_API_KEY="wrong").status_code, 401)

    def test_upload_and_at_risk_list(self):
        csv = settings.SAMPLE_DATA.read_bytes()
        resp = self.client.post("/api/upload", {"file": SimpleUploadedFile("s.csv", csv)})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["created"], 395)
        flagged = self.client.get("/api/students/at-risk?limit=5").json()
        self.assertEqual(len(flagged), 5)
        probs = [s["risk_probability"] for s in flagged]
        self.assertEqual(probs, sorted(probs, reverse=True))


@skipUnless(TRAINED, "model not trained")
class PageTests(BaseCase):
    def test_pages_render(self):
        import_dataframe(read_table(settings.SAMPLE_DATA))
        for url in ["/", "/students/", "/students/?risk=flagged&q=1", "/students/add/",
                    "/upload/", "/metrics/", "/students/1/"]:
            self.assertEqual(self.client.get(url).status_code, 200, url)

    def test_login_required(self):
        self.client.logout()
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/admin/login/", resp["Location"])

    def test_add_student_through_form_scores_it(self):
        resp = self.client.post("/students/add/", {**GOOD, "name": "Test", "G1": 4, "G2": 3})
        self.assertEqual(resp.status_code, 302)
        s = Student.objects.get(name="Test")
        self.assertEqual(s.model_used, "full")
        self.assertTrue(s.at_risk)
