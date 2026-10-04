from django.contrib import admin, messages

from . import services
from .models import Student


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ("display_name", "sex", "age", "absences", "failures",
                    "risk_probability", "risk_level", "at_risk", "model_used")
    list_filter = ("risk_level", "at_risk", "sex", "model_used")
    search_fields = ("name",)
    readonly_fields = ("risk_probability", "at_risk", "risk_level", "model_used",
                       "predicted_at", "created_at")
    actions = ["rescore"]

    @admin.action(description="Re-run risk prediction for selected students")
    def rescore(self, request, queryset):
        students = list(queryset)
        try:
            services.apply_predictions(students)
        except services.ModelNotTrained as exc:
            self.message_user(request, str(exc), messages.ERROR)
            return
        Student.objects.bulk_update(
            students, ["risk_probability", "at_risk", "risk_level", "model_used", "predicted_at"])
        self.message_user(request, f"Re-scored {len(students)} students.")

    def save_model(self, request, obj, form, change):
        try:
            services.apply_predictions([obj])
        except services.ModelNotTrained:
            pass
        super().save_model(request, obj, form, change)
