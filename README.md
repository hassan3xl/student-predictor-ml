# 🎓 Student Academic Performance Predictor

> An early-warning machine learning web application built with **Django** and **scikit-learn** that identifies students at risk of academic failure before it is too late to intervene.

---

## 📌 Table of Contents
1. [What is This Application?](#what-is-this-application)
2. [Absolute Beginner's Quickstart (Step-by-Step)](#absolute-beginners-quickstart-step-by-step)
   - [Prerequisites](#prerequisites)
   - [Step 1: Clone the Repository](#step-1-clone-the-repository)
   - [Step 2: Create and Activate a Virtual Environment](#step-2-create-and-activate-a-virtual-environment)
   - [Step 3: Install Required Dependencies](#step-3-install-required-dependencies)
   - [Step 4: Train / Initialize the Machine Learning Models](#step-4-train--initialize-the-machine-learning-models)
   - [Step 5: Apply Database Migrations](#step-5-apply-database-migrations)
   - [Step 6: Create a Staff / Admin Account](#step-6-create-a-staff--admin-account)
   - [Step 7: Launch the Application](#step-7-launch-the-application)
   - [Step 8: (Optional) Run the Test Suite](#step-8-optional-run-the-test-suite)
3. [Understanding the Database (`db.sqlite3`)](#understanding-the-database-dbsqlite3)
   - [Why the SQLite Database is Included](#why-the-sqlite-database-is-included)
   - [How to Reset the Database or Use Your Own Data](#how-to-reset-the-database-or-use-your-own-data)
4. [How Machine Learning (ML) Works in This Project](#how-machine-learning-ml-works-in-this-project)
   - [The Problem We Are Solving](#the-problem-we-are-solving)
   - [What Information Does the AI Look At? (Features)](#what-information-does-the-ai-look-at-features)
   - [The Two-Model Strategy: Early-Warning vs. Full Model](#the-two-model-strategy-early-warning-vs-full-model)
   - [How the AI Learns: Preprocessing, Tournaments & Thresholds](#how-the-ai-learns-preprocessing-tournaments--thresholds)
   - [Why We Care About "Recall" More Than Simple Accuracy](#why-we-care-about-recall-more-than-simple-accuracy)
5. [Project Tour: What Every File and Folder Does](#project-tour-what-every-file-and-folder-does)
6. [Web Pages and User Interface Guide](#web-pages-and-user-interface-guide)
7. [REST API (For Developers & External Systems)](#rest-api-for-developers--external-systems)
8. [Using Your Own School's Data](#using-your-own-schools-data)

---

## What is This Application?

In many schools and universities, students who struggle academically are only identified after they fail their final exams. By then, it is often too late to provide tutoring, counseling, or academic interventions.

This project solves that problem. It is a full-featured web portal designed for teachers, counselors, and school administrators:
- **Web Portal:** Teachers can log in, view interactive dashboards, search student profiles, add new students, and import spreadsheets (CSVs).
- **Machine Learning Engine:** Analyzes background factors (study time, family support, absences, past failures, etc.) and calculates the exact statistical probability that a student will score below passing grade (less than 10 out of 20).
- **Two-Stage Prediction:** Can alert staff **before any exams happen** (Early-Warning Model) or provide a refined forecast once preliminary grades arrive (Full Model).
- **REST API:** Includes a built-in API (powered by Django Ninja) so student information systems (SIS) or mobile apps can submit student data and receive risk scores automatically.

---

## Absolute Beginner's Quickstart (Step-by-Step)

If you have never used **Django** (a Python framework for building web apps) or **Machine Learning** before, follow these instructions step-by-step. Everything you need to type into your computer terminal or command prompt is clearly shown.

### Prerequisites

Before starting, ensure your computer has:
1. **Python** (version 3.10, 3.11, 3.12, or 3.13). You can check by opening your terminal and typing `python3 --version` or `python --version`.
2. **Git** (version control tool to download the project).

---

### Step 1: Clone the Repository

Download the code to your computer by opening your terminal (macOS/Linux) or Command Prompt / PowerShell (Windows) and running:

```bash
git clone https://github.com/hassan3xl/student-predictor-ml.git
cd student_predictor
```
*(If you downloaded a `.zip` file instead, extract it into a folder, open your terminal, and `cd` into that folder).*

---

### Step 2: Create and Activate a Virtual Environment

A **virtual environment** is a self-contained, isolated folder on your computer where Python installs all the libraries needed specifically for this project without messing up the rest of your system.

#### On macOS and Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows (Command Prompt):
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### On Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

> **How to tell it worked:** Your terminal prompt will now show `(venv)` at the very beginning of the line!

---

### Step 3: Install Required Dependencies

The file `requirements.txt` lists all the Python software packages this project needs (such as Django for the website, scikit-learn for machine learning, pandas for handling spreadsheets, and matplotlib for charts).

Install all of them with one command:

```bash
pip install -r requirements.txt
```

*(This may take 1 to 2 minutes while pip downloads and installs the libraries).*

---

### Step 4: Train / Initialize the Machine Learning Models

Machine learning models are saved as binary files (`.joblib`). Because these files depend on the exact version of Python and scikit-learn installed on your computer, **you should always run the training command once** after installing dependencies:

```bash
python manage.py train_model
```

#### What happens during this step:
- It loads the training dataset (`data/student-mat.csv`).
- It cleans and normalizes the data.
- It tests multiple algorithms (Logistic Regression, Random Forest, Gradient Boosting) using cross-validation.
- It selects the top-performing model, tunes the alert threshold, and saves the ready-to-use models into the `ml_artifacts/` folder.
- It generates diagnostic visual charts in `predictor/static/predictor/img/`.
- **Time required:** Approximately 20–30 seconds.

---

### Step 5: Apply Database Migrations

Django needs to set up the database tables (e.g., student tables, user login tables). Run:

```bash
python manage.py migrate
```

If the database file already exists, Django will verify all tables are up-to-date in a couple of seconds.

---

### Step 6: Create a Staff / Admin Account

To access the dashboard and view student details, you need a login account. Create your personal administrator account by running:

```bash
python manage.py createsuperuser
```

You will be prompted to:
1. Choose a **Username** (for example: `admin` or your name)
2. Enter an **Email address** (can be left blank or `admin@example.com`)
3. Enter a **Password** (note: typing the password will not show characters on screen for security; just type it and press Enter)
4. Confirm the **Password**

---

### Step 7: Launch the Application

Now you are ready to start the web server! Run:

```bash
python manage.py runserver
```

You will see output similar to:
```text
Watching for file changes with StatReloader
Performing system checks...
System check identified no issues (0 silenced).
Django version 6.1, using settings 'config.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.
```

1. Open your web browser (Chrome, Firefox, Edge, Safari).
2. Visit: **`http://127.0.0.1:8000/`**
3. Log in with the **Username** and **Password** you created in Step 6.
4. You will land on the interactive **Student Risk Dashboard**!

> **To stop the server:** Press `Ctrl + C` in your terminal at any time.

---

### Step 8: (Optional) Run the Test Suite

The project includes an automated test suite verifying data import, validation, API endpoints, web views, and machine learning scoring:

```bash
python manage.py test predictor
```

All 12 tests should finish with `OK`.

---

## Understanding the Database (`db.sqlite3`)

### Why the SQLite Database is Included

In traditional web development, setting up a database often requires installing heavy database software (like PostgreSQL or MySQL), creating database users, configuring ports, and setting passwords.

To make this project **100% plug-and-play for everyone**, it uses **SQLite**. SQLite stores the entire database inside a single local file called **`db.sqlite3`**.

- **Bundled Data:** The repository includes `db.sqlite3` pre-loaded with **395 student records** from the UCI Student Performance dataset, with all predictions already computed.
- **Immediate Experience:** Anyone who clones this project can immediately see a populated dashboard with charts, risk levels, and student profiles without needing to import data first.

---

### How to Reset the Database or Use Your Own Data

If you do **not** want the pre-loaded sample data and want a completely clean database or want to upload your school's own dataset, you have three simple options:

#### Option A: Quick Clear via Command Line (Keep Admin User)
If you want to keep your user account but erase all 395 sample students and optionally load your own CSV file:

```bash
# Erase existing students and import your CSV:
python manage.py import_students --clear --file path/to/your_students.csv

# Or simply erase all students without importing anything:
python manage.py shell -c "from predictor.models import Student; Student.objects.all().delete()"
```

#### Option B: Complete Clean Slate (Delete and Recreate)
If you want to completely destroy the database and start from zero:

1. **Delete the database file:**
   - **Linux / macOS:** `rm db.sqlite3`
   - **Windows:** `del db.sqlite3`
2. **Re-create empty database tables:**
   ```bash
   python manage.py migrate
   ```
3. **Create your new administrator account:**
   ```bash
   python manage.py createsuperuser
   ```
4. **Import your own students:**
   Use the web interface at `http://127.0.0.1:8000/upload/` or run:
   ```bash
   python manage.py import_students --file path/to/your_students.csv
   ```

#### Option C: Upload via Web Interface
You can also import student records through the browser:
1. Log in to `http://127.0.0.1:8000/`
2. Click **Upload Data** in the top navigation bar (or visit `/upload/`).
3. Select your CSV file and click **Upload & Predict**.
4. The application validates every row, highlights any errors, saves valid students, and scores them automatically!

---

## How Machine Learning (ML) Works in This Project

> **No ML background? No problem!** Here is a plain-English explanation of how the artificial intelligence works under the hood.

### The Problem We Are Solving

In traditional computer programming, a developer writes manual rules, like:
> *If absences > 10 and failures > 2, then flag student.*

Real life is much more nuanced: a student with 10 absences might still pass if they have strong family support and study 8 hours a week, while another student with only 4 absences might fail due to lack of study time and high alcohol consumption.

**Machine Learning** does not use fixed manual rules. Instead, it looks at hundreds of past student records where the final grade was already known. The algorithm finds subtle mathematical correlations and patterns, and builds a predictive model that outputs a **risk probability** between `0.0` (0% chance of failing) and `1.0` (100% chance of failing).

In this project, a student is considered **"At Risk"** if their final grade ($G3$) is projected to be **below 10 out of 20** (the standard pass mark).

---

### What Information Does the AI Look At? (Features)

The model evaluates four key categories of student information (called *features*):

| Category | What it includes | Examples |
|---|---|---|
| **Demographics** | Basic student background | Age, sex, urban vs. rural home address, family size, parents living together or apart |
| **Family Background** | Educational and household context | Mother's & Father's education levels (0 to 4), Mother's & Father's jobs, guardian, family relationship quality (1 to 5) |
| **Study Habits & School** | Academic commitment & support | Weekly study time (1 to 4), travel time, past class failures (0 to 4), extra school support, paid tutoring, extra-curriculars, nursery school, plans for higher education, home internet, romantic relationship |
| **Lifestyle & Health** | Daily habits and well-being | Free time after school, going out with friends, workday & weekend alcohol use (1 to 5), health status (1 to 5), school absences (0 to 100) |
| **Period Grades (Optional)** | Early academic progress | First period exam mark ($G1$, 0–20), Second period exam mark ($G2$, 0–20) |

---

### The Two-Model Strategy: Early-Warning vs. Full Model

A major real-world challenge in education is: **at what point in the academic year are you evaluating the student?**

To solve this, this project trains and deploys **two separate models**:

```
                       ┌─────────────────────────┐
                       │   New Student Record    │
                       └────────────┬────────────┘
                                    │
                         Are G1 and G2 grades
                         entered for this student?
                                    │
                   ┌────────────────┴────────────────┐
                 NO│                                 │YES
                   ▼                                 ▼
       ┌────────────────────────┐       ┌────────────────────────┐
       │   Early-Warning Model  │       │       Full Model       │
       │   (model_early.joblib) │       │   (model_full.joblib)  │
       ├────────────────────────┤       ├────────────────────────┤
       │ Uses background, habits│       │ Uses all background    │
       │ & absences only.       │       │ PLUS G1 & G2 grades.   │
       │                        │       │                        │
       │ Best for: Start of the │       │ Best for: Mid-year     │
       │ term, before any tests.│       │ once term marks exist. │
       └────────────────────────┘       └────────────────────────┘
```

1. **Early-Warning Model (`model_early.joblib`)**:
   - Uses **30 features** (background, habits, absences, past failures).
   - **Does NOT require any test grades.**
   - Ideal for the very beginning of the school year to identify students who need support on day one.
2. **Full / Grades Model (`model_full.joblib`)**:
   - Uses **32 features** (all 30 background features plus $G1$ and $G2$ term grades).
   - Once students take their first and second period exams, their past exam performance is a powerful indicator of final exam results. This model achieves very high accuracy (~91%).

**Smart Automatic Selection:**
You don't need to manually pick which model to use! Whenever a student is created or scored, `predictor/services.py` checks if $G1$ and $G2$ exist:
- If both are present $\rightarrow$ uses the **Full Model**.
- If either is missing or blank $\rightarrow$ automatically falls back to the **Early-Warning Model**.

---

### How the AI Learns: Preprocessing, Tournaments & Thresholds

When you run `python manage.py train_model`, here is what happens step-by-step inside `predictor/training.py` and `predictor/mlkit.py`:

#### 1. Data Cleaning & Preprocessing (The Scikit-Learn Pipeline)
Computers can only do math with clean numbers. Raw data often has text, missing fields, or abnormal spikes. We wrap everything in an automated pipeline:
- **Missing Value Imputation (`SimpleImputer`):** If a numerical value is missing, it is filled with the column's *median*. If a category is missing, it is filled with the *most frequent value*.
- **Outlier Clipping (`IQRClipper`):** If a student has an extreme anomaly (e.g. 75 absences when normal is 4), extreme numbers can distort the model. The custom `IQRClipper` clips extreme values to a safe boundary using the Interquartile Range (IQR).
- **Feature Scaling (`StandardScaler`):** Normalizes numeric numbers (e.g., age 15–22, absences 0–93) so that large scales don't overpower small scales.
- **One-Hot Encoding (`OneHotEncoder`):** Converts text categories (like `Mjob="teacher"`, `Mjob="health"`) into numerical columns ($1$s and $0$s) that algorithms can process.

#### 2. The Model Tournament (Algorithm Comparison)
The training script compares three prominent machine learning algorithms:
1. **Logistic Regression:** A linear statistical model that calculates the probability of an event using a sigmoid curve.
2. **Random Forest:** An ensemble of hundreds of decision trees that each look at different subsets of features and vote.
3. **Gradient Boosting:** An ensemble that builds decision trees sequentially, with each new tree focusing specifically on correcting the mistakes of previous trees.

To ensure fair evaluation, the dataset is split:
- **80% Training Set:** Used to train and compare models using **Repeated Stratified 5-Fold Cross-Validation** (the data is split into 5 parts, trained on 4, tested on 1, and repeated 3 times).
- **20% Hold-Out Test Set:** Completely hidden during training, used only at the end to verify how well the winning model performs on new, unseen data.
- The algorithm with the highest **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve) wins.

---

### Why We Care About "Recall" More Than Simple Accuracy

In standard machine learning, models often use a default `0.50` (50%) probability threshold: if risk $\ge 50\%$, flag the student; otherwise, mark as safe.

However, **in school interventions, not all mistakes are equal:**
- **False Alarm (False Positive):** The model flags a student who would have passed anyway. *Consequence:* The teacher checks in on the student, offering a bit of extra support. Minimal harm done.
- **Missed Student (False Negative):** The model says a student is safe, but they actually fail. *Consequence:* The student slips through the cracks without getting help. **Very bad outcome!**

For this reason, our system calibrates the **Alert Threshold**:
- It searches for the threshold that catches **at least 80% of all truly at-risk students** (achieving $\ge 80\%$ **Recall**).
- It then assigns students to intuitive risk bands:
  - 🟢 **Low Risk:** Probability below $60\%$ of the threshold.
  - 🟡 **Medium Risk:** Probability between $60\%$ and $100\%$ of the threshold.
  - 🔴 **High Risk / Flagged:** Probability equal to or above the threshold.

---

## Project Tour: What Every File and Folder Does

Here is an architectural map of the project files and what each one does:

```
student_predictor/
│
├── manage.py                     # Django's master terminal CLI tool
├── db.sqlite3                    # The pre-populated SQLite database file
├── requirements.txt              # List of Python dependencies
├── pyproject.toml / uv.lock      # Project metadata and locked dependency specifications
├── main.py                       # Helper entry point
│
├── config/                       # Project Configuration (The Django Brain)
│   ├── settings.py               # Main settings: database, apps, security, paths
│   ├── urls.py                   # Master URL routing table (routes /admin, /api, and app views)
│   ├── wsgi.py                   # Web Server Gateway Interface for production deployments
│   └── __init__.py               # Marks config as a Python package
│
├── data/                         # Raw Data Storage
│   └── student-mat.csv           # Bundled UCI benchmark dataset (Portuguese maths students)
│
├── ml_artifacts/                 # Output of Model Training (Saved Binaries)
│   ├── model_early.joblib        # Trained scikit-learn pipeline for early warning (no grades)
│   ├── model_full.joblib         # Trained scikit-learn pipeline including G1 & G2 grades
│   └── metrics.json              # Full training metrics, thresholds, accuracy, and confusion matrix data
│
└── predictor/                    # The Main Application Module
    ├── models.py                 # Database Schema: Defines the Student model and fields
    ├── mlkit.py                  # Pure Machine Learning tools: features list, IQRClipper, pipelines
    ├── training.py               # Offline training engine: runs cross-validation, tunes thresholds, plots charts
    ├── services.py               # The bridge: loads models into memory and computes predictions
    ├── views.py                  # Web page controllers: handles Dashboard, Lists, Forms, Uploads
    ├── forms.py                  # HTML Form validation for adding students and uploading CSVs
    ├── importer.py               # Validates CSV rows, creates Student objects, and triggers scoring
    ├── urls.py                   # Web page URL routes (/students, /upload, /metrics, etc.)
    ├── api.py                    # Django Ninja REST API endpoints (/api/predict, /api/students)
    ├── admin.py                  # Django Admin custom interface with bulk prediction tools
    ├── tests.py                  # Automated test suite (12 unit and integration tests)
    │
    ├── management/commands/      # Custom Terminal Commands
    │   ├── train_model.py        # Management command: `python manage.py train_model`
    │   └── import_students.py    # Management command: `python manage.py import_students`
    │
    ├── templates/predictor/      # HTML User Interface Templates
    │   ├── base.html             # Common navigation bar, head, and styling layout
    │   ├── dashboard.html        # Main dashboard with risk breakdown, charts, and top-risk list
    │   ├── student_list.html     # Searchable, filterable list of all students
    │   ├── student_detail.html   # Detailed profile of a single student showing prediction factors
    │   ├── student_form.html     # Form for adding a new student manually
    │   ├── upload.html           # Drag-and-drop CSV upload page with validation feedback
    │   └── metrics.html          # Performance dashboard showing cross-validation tables & plots
    │
    └── static/predictor/         # Static Assets & Generated Plots
        └── img/                  # Generated diagnostic plots (confusion matrices, feature importance)
            ├── early_confusion.png
            ├── early_importance.png
            ├── full_confusion.png
            └── full_importance.png
```

---

## Web Pages and User Interface Guide

When running the web server (`python manage.py runserver`), you can access these pages:

| Page URL | Name | What You Can Do There |
|---|---|---|
| `/` | **Dashboard** | Overview of all students: total count, percentage flagged at risk, risk level distribution chart, probability score histogram, and the Top 10 highest-risk students needing immediate attention. |
| `/students/` | **Student Directory** | Searchable and filterable registry. Filter by risk status (Flagged, High, Medium, Low, or Unscored) or search by student name / ID. |
| `/students/add/` | **Add Student** | A clean, grouped web form to enter a student's demographics, family, study habits, lifestyle, and optional grades. Saves the record and immediately calculates their risk score. |
| `/students/<id>/` | **Student Profile** | Displays the student's predicted risk score, risk level badge, which model was used (early vs. full), threshold applied, full background breakdown, and a button to re-run predictions. |
| `/upload/` | **Upload CSV** | Upload a spreadsheet of student records. Validates every row individually, displays an error report for invalid rows, and automatically scores valid rows. |
| `/metrics/` | **Model Evaluation** | Full transparency page: displays data quality stats, cross-validation comparison tables, hold-out test set accuracy/precision/recall/ROC-AUC, confusion matrices, and feature importance bar graphs. |
| `/admin/` | **Django Admin** | Standard Django administrator panel. Allows manual editing of any record and bulk actions (like selecting 50 students and clicking "Re-run prediction"). |

---

## REST API (For Developers & External Systems)

The project includes a built-in REST API powered by **Django Ninja**.

### Interactive API Documentation
Visit **`http://127.0.0.1:8000/api/docs`** in your browser to view an interactive Swagger UI where you can test any API endpoint live.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Sends student data in JSON format; returns probability, risk level, and threshold (does not save to database). |
| `GET` | `/api/students` | Returns a list of students with pagination (`?limit=50&offset=0`). |
| `GET` | `/api/students/at-risk` | Returns only flagged students ordered by highest risk probability first (`?min_probability=0.6`). |
| `GET` | `/api/students/{id}` | Returns a single student's record and prediction results. |
| `POST` | `/api/upload` | Multipart file upload endpoint to import a CSV programmatically. |
| `GET` | `/api/metrics` | Returns JSON of the active machine learning models' performance and thresholds. |

### API Authentication
The API can be accessed in two ways:
1. **Browser Session:** If you are already logged in via the web browser, API calls work automatically.
2. **API Key Header (`X-API-Key`):** Set the `API_KEY` environment variable when running the server:
   ```bash
   API_KEY="my-secret-key" python manage.py runserver
   ```
   External clients can then send requests with the header:
   ```bash
   curl -H "X-API-Key: my-secret-key" "http://127.0.0.1:8000/api/students/at-risk?limit=5"
   ```

---

## Using Your Own School's Data

The bundled dataset (`data/student-mat.csv`) is based on the famous Portuguese secondary school mathematics study. When you are ready to use your own school's historical records:

### 1. Prepare Your CSV File
Ensure your CSV contains the feature columns (see the list in `/upload/` or `predictor/mlkit.py`) and a `G3` column representing the final grade.

### 2. Retrain the Models on Your Data
Run the training command pointing to your CSV file:
```bash
python manage.py train_model --data path/to/your_school_data.csv
```

### 3. Import Students into the Database
Clear out sample data and load your new students:
```bash
python manage.py import_students --file path/to/your_school_data.csv --clear
```

### 4. Customizing the Passing Grade
By default, the pass mark is 10 out of 20 (`PASS_MARK = 10` in `predictor/mlkit.py`). If your school uses a different grading scale (such as percentage where 50 or 60 is a pass, or a 4.0 GPA scale):
1. Open `predictor/mlkit.py`.
2. Change `PASS_MARK = 10` to your desired passing cutoff.
3. Re-run `python manage.py train_model`.

---

## Summary of Useful Commands

| Task | Command |
|---|---|
| Activate virtual environment (Linux/macOS) | `source venv/bin/activate` |
| Activate virtual environment (Windows) | `venv\Scripts\activate` |
| Install dependencies | `pip install -r requirements.txt` |
| Train ML models | `python manage.py train_model` |
| Update database schema | `python manage.py migrate` |
| Create admin user | `python manage.py createsuperuser` |
| Start web server | `python manage.py runserver` |
| Run automated tests | `python manage.py test predictor` |
| Import students from CSV | `python manage.py import_students --file your_file.csv` |
| Reset & clear all students | `python manage.py import_students --clear --file your_file.csv` |
