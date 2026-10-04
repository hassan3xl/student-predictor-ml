from django.urls import path

from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("students/", views.student_list, name="student_list"),
    path("students/add/", views.student_add, name="student_add"),
    path("students/<int:pk>/", views.student_detail, name="student_detail"),
    path("students/<int:pk>/predict/", views.student_predict, name="student_predict"),
    path("predict-all/", views.predict_all, name="predict_all"),
    path("upload/", views.upload, name="upload"),
    path("metrics/", views.metrics_page, name="metrics"),
]
