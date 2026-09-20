import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CardioCare AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL UI STYLING - VISUAL ONLY
# ============================================================

st.markdown(r"""
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1450px; }
section[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,0.18); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; padding-left: 1.1rem; padding-right: 1.1rem; }
h1 { letter-spacing: -0.5px; }
h2 { margin-top: 0.8rem; }
h3 { margin-top: 0.5rem; }
div[data-testid="stMetric"] { padding: 1rem 1.1rem; border: 1px solid rgba(128,128,128,0.18); border-radius: 12px; background: rgba(128,128,128,0.035); }
div.stButton > button { border-radius: 9px; min-height: 2.6rem; font-weight: 600; }
button[data-baseweb="tab"] { font-weight: 600; }
div[data-testid="stExpander"] { border-radius: 10px; border: 1px solid rgba(128,128,128,0.18); }
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
div[data-testid="stAlert"] { border-radius: 10px; }
.cardiocare-hero { padding: 1.35rem 1.5rem; border: 1px solid rgba(128,128,128,0.18); border-radius: 15px; background: rgba(128,128,128,0.035); margin-bottom: 1.25rem; }
.cardiocare-hero-title { font-size: 2rem; font-weight: 700; margin: 0; }
.cardiocare-hero-text { margin: 0.4rem 0 0 0; opacity: 0.72; }
.workspace-card { min-height: 155px; padding: 1.1rem; border: 1px solid rgba(128,128,128,0.18); border-radius: 12px; background: rgba(128,128,128,0.025); }
.workspace-card h3 { margin: 0 0 0.55rem 0; }
.workspace-card p { margin: 0; opacity: 0.72; line-height: 1.5; }
.section-label { font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.62; margin-bottom: 0.45rem; }
.evaluation-header { padding: 1rem 1.2rem; border: 1px solid rgba(128,128,128,0.18); border-radius: 12px; background: rgba(128,128,128,0.025); margin-bottom: 1rem; }
@media (max-width: 900px) { .block-container { padding-left: 1rem; padding-right: 1rem; } .cardiocare-hero-title { font-size: 1.55rem; } }
</style>
""", unsafe_allow_html=True)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

DATA_FILE = "cardio_without_scale.csv"

FEATURES = [
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active"
]

TARGET = "cardio"

RANDOM_STATE = 42

TUNED_MODEL_DIR = Path("tuned_models")
TUNED_MODEL_DIR.mkdir(exist_ok=True)

TUNED_MODEL_FILES = {
    "Logistic Regression": TUNED_MODEL_DIR / "logistic_regression_tuned.pkl",
    "Decision Tree": TUNED_MODEL_DIR / "decision_tree_tuned.pkl",
    "KNN": TUNED_MODEL_DIR / "knn_tuned.pkl",
    "Random Forest": TUNED_MODEL_DIR / "random_forest_tuned.pkl",
    "AdaBoost": TUNED_MODEL_DIR / "adaboost_tuned.pkl",
}


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_FILE)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    data = data[required_columns].dropna()

    return data


try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "cardio_without_scale.csv was not found."
    )

    st.info(
        "Place cardio_without_scale.csv in the same "
        "folder as this Streamlit Python file."
    )

    st.stop()

except Exception as error:

    st.error(
        f"Dataset error: {error}"
    )

    st.stop()


X = df[FEATURES]
y = df[TARGET]


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(model_name):

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    if model_name == "Logistic Regression":

        return Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=RANDOM_STATE
                    )
                )
            ]
        )


    # --------------------------------------------------------
    # Decision Tree
    # --------------------------------------------------------

    elif model_name == "Decision Tree":

        return DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=RANDOM_STATE
        )


    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------

    elif model_name == "KNN":

        return Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "knn",
                    KNeighborsClassifier(
                        n_neighbors=25,
                        weights="distance",
                        n_jobs=-1
                    )
                )
            ]
        )


    # --------------------------------------------------------
    # Random Forest
    # Tuned parameters obtained from your tuning
    # --------------------------------------------------------

    elif model_name == "Random Forest":

        return RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )


    # --------------------------------------------------------
    # AdaBoost
    # Tuned parameters obtained from your tuning
    # --------------------------------------------------------

    elif model_name == "AdaBoost":

        return AdaBoostClassifier(
            n_estimators=150,
            learning_rate=0.5,
            estimator=DecisionTreeClassifier(
                max_depth=3,
                random_state=RANDOM_STATE
            ),
            random_state=RANDOM_STATE
        )


# ============================================================
# HYPERPARAMETER TUNING - RANDOMIZED SEARCH
# ============================================================

def get_tuning_configuration(model_name):
    if model_name == "Logistic Regression":
        estimator = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
        params = {"model__C": [0.01, 0.1, 1, 10, 100], "model__solver": ["liblinear", "lbfgs"], "model__class_weight": [None, "balanced"]}
    elif model_name == "Decision Tree":
        estimator = DecisionTreeClassifier(random_state=RANDOM_STATE)
        params = {"max_depth": [3, 5, 6, 8, 10, 12, 15, 20], "min_samples_split": [2, 5, 10, 20], "min_samples_leaf": [1, 2, 5, 10], "max_features": [None, "sqrt", "log2"], "criterion": ["gini", "entropy"]}
    elif model_name == "KNN":
        estimator = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_jobs=-1))])
        params = {"knn__n_neighbors": [5, 10, 15, 20, 25, 30, 35, 40], "knn__weights": ["uniform", "distance"], "knn__p": [1, 2]}
    elif model_name == "Random Forest":
        estimator = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        params = {"n_estimators": [100, 150, 200], "max_depth": [10, 15, 20], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2], "max_features": ["sqrt", "log2"]}
    elif model_name == "AdaBoost":
        estimator = AdaBoostClassifier(random_state=RANDOM_STATE)
        params = {"n_estimators": [50, 100, 150, 200], "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0], "estimator": [DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE), DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE), DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)]}
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return estimator, params


def tune_model(model_name, n_iter=8, folds=3):
    model_file = TUNED_MODEL_FILES[model_name]
    if model_file.exists():
        try:
            return joblib.load(model_file)
        except Exception:
            model_file.unlink(missing_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    estimator, params = get_tuning_configuration(model_name)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(estimator, param_distributions=params, n_iter=n_iter, cv=cv, scoring="accuracy", random_state=RANDOM_STATE, n_jobs=-1, verbose=1, refit=True)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    pred = model.predict(X_test)
    result = {
        "model": model_name,
        "best_params": search.best_params_,
        "best_cv_accuracy": search.best_score_,
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred, zero_division=0),
        "test_recall": recall_score(y_test, pred, zero_division=0),
        "test_f1": f1_score(y_test, pred, zero_division=0),
    }
    model._cardiocare_tuning_result_ = result
    joblib.dump(model, model_file, compress=3)
    return model


def tune_all_models():
    results = []
    progress = st.progress(0)
    status = st.empty()
    names = list(TUNED_MODEL_FILES.keys())
    for i, name in enumerate(names, 1):
        status.info(f"Tuning {name} ({i}/{len(names)})...")
        model = tune_model(name, n_iter=8, folds=3)
        results.append(model._cardiocare_tuning_result_)
        progress.progress(i / len(names))
    status.success("All tuned models are saved.")
    return pd.DataFrame(results)


# ============================================================
# HOLDOUT EVALUATION
# ============================================================

@st.cache_data
def evaluate_holdout(model_name, test_size):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = create_model(model_name)

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(X_test)

    probability = model.predict_proba(
        X_test
    )[:, 1]

    return (
        y_test,
        prediction,
        probability
    )


# ============================================================
# CROSS VALIDATION
# ============================================================

@st.cache_data
def evaluate_cv(model_name, folds):

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    model = create_model(model_name)

    # --------------------------------------------------------
    # IMPORTANT:
    # Only ONE cross_val_predict call.
    #
    # method="predict_proba" gives us probabilities.
    # We derive predictions from probability.
    #
    # This is much faster than:
    # cross_val_predict(predict)
    # cross_val_predict(predict_proba)
    # cross_validate()
    # --------------------------------------------------------

    probability = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    prediction = (
        probability >= 0.5
    ).astype(int)

    return (
        y,
        prediction,
        probability
    )


# ============================================================
# BOOTSTRAP EVALUATION
# ============================================================

@st.cache_data
def evaluate_bootstrap(model_name, iterations):

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    number_of_rows = len(X)

    metric_results = []

    all_actual = []
    all_prediction = []
    all_probability = []

    for iteration in range(iterations):

        bootstrap_indices = rng.choice(
            number_of_rows,
            size=number_of_rows,
            replace=True
        )

        unique_indices = np.unique(
            bootstrap_indices
        )

        oob_indices = np.setdiff1d(
            np.arange(number_of_rows),
            unique_indices
        )

        if len(oob_indices) == 0:
            continue

        X_train = X.iloc[
            bootstrap_indices
        ]

        y_train = y.iloc[
            bootstrap_indices
        ]

        X_test = X.iloc[
            oob_indices
        ]

        y_test = y.iloc[
            oob_indices
        ]

        model = create_model(
            model_name
        )

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

        probability = model.predict_proba(
            X_test
        )[:, 1]

        metric_results.append(
            {
                "Accuracy":
                    accuracy_score(
                        y_test,
                        prediction
                    ),

                "Precision":
                    precision_score(
                        y_test,
                        prediction,
                        zero_division=0
                    ),

                "Recall":
                    recall_score(
                        y_test,
                        prediction,
                        zero_division=0
                    ),

                "F1 Score":
                    f1_score(
                        y_test,
                        prediction,
                        zero_division=0
                    ),

                "ROC-AUC":
                    roc_auc_score(
                        y_test,
                        probability
                    )
            }
        )

        all_actual.extend(
            y_test.tolist()
        )

        all_prediction.extend(
            prediction.tolist()
        )

        all_probability.extend(
            probability.tolist()
        )

    return (
        pd.DataFrame(metric_results),
        np.array(all_actual),
        np.array(all_prediction),
        np.array(all_probability)
    )


# ============================================================
# FINAL DEPLOYMENT MODEL
# ============================================================

@st.cache_resource
def train_final_model(model_name):

    model = create_model(
        model_name
    )

    model.fit(
        X,
        y
    )

    return model


# ============================================================

# ============================================================
# SHARED SIDEBAR CONFIGURATION
# ============================================================

with st.sidebar:
    st.title("❤️ CardioCare")
    st.caption("AI Cardiovascular Risk Prediction")
    st.divider()
    st.subheader("Configuration")
    selected_model = st.selectbox(
        "Machine Learning Model",
        ["Logistic Regression", "Decision Tree", "KNN", "Random Forest", "AdaBoost"]
    )
    selected_method = st.selectbox(
        "Evaluation Method",
        ["Holdout", "Cross-Validation", "Bootstrap"]
    )
    st.divider()
    if selected_method == "Holdout":
        test_percentage = st.slider("Testing Data (%)", 10, 40, 20, 5)
        test_size = test_percentage / 100
    elif selected_method == "Cross-Validation":
        folds = st.selectbox("Number of Folds", [3, 5])
        st.caption("3 folds is faster. 5 folds gives a more detailed evaluation.")
    else:
        bootstrap_iterations = st.slider("Bootstrap Iterations", 5, 20, 10, 5)
        st.caption("Fewer iterations are used to keep the application fast.")
    st.divider()
    st.subheader("Dataset")
    st.metric("Records", f"{len(df):,}")
    st.metric("Features", len(FEATURES))
    st.metric("Target Classes", y.nunique())
    st.divider()
    st.caption("Version 3.0 • Machine Learning Project")

# ============================================================
# SESSION STATE
# ============================================================
if "tuning_result" not in st.session_state:
    st.session_state.tuning_result = None
if "tuning_all_result" not in st.session_state:
    st.session_state.tuning_all_result = None


def dashboard_page():
    st.markdown(r"""
    <div class="cardiocare-hero">
        <div class="cardiocare-hero-title">❤️ CardioCare AI</div>
        <p class="cardiocare-hero-text">A clean workspace for cardiovascular disease prediction and machine-learning analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("Selected Model", selected_model)
    c2.metric("Evaluation", selected_method)
    c3.metric("Patients", f"{len(df):,}")
    c4.metric("Features", len(FEATURES))

    st.subheader("📌 Workspace")
    a, b, c = st.columns(3, gap="medium")
    with a:
        st.markdown(r"""<div class="workspace-card"><h3>📊 Prediction & Evaluation</h3><p>Enter patient information, generate a prediction, and inspect model performance using Holdout, Cross-Validation, or Bootstrap.</p></div>""", unsafe_allow_html=True)
    with b:
        st.markdown(r"""<div class="workspace-card"><h3>⚙️ Hyperparameter Tuning</h3><p>Optimize model settings with RandomizedSearchCV and save tuned models for later use.</p></div>""", unsafe_allow_html=True)
    with c:
        st.markdown(r"""<div class="workspace-card"><h3>🤖 Models</h3><p>Review Logistic Regression, Decision Tree, KNN, Random Forest, and AdaBoost model information.</p></div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("🤖 Current Model Information")
    descriptions = {
        "Logistic Regression": "StandardScaler → Logistic Regression",
        "Decision Tree": "Controlled depth, minimum split and minimum leaf size.",
        "KNN": "StandardScaler → KNN with k=25 and distance weighting.",
        "Random Forest": "Random Forest with the configured ensemble parameters.",
        "AdaBoost": "AdaBoost with the configured boosting parameters."
    }
    st.info(descriptions[selected_model])
    st.subheader("📁 Dataset Overview")
    st.dataframe(df[FEATURES].describe().T.round(2), use_container_width=True)


def prediction_and_evaluation_page():
    st.markdown(r"""
    <div class="cardiocare-hero">
        <div class="cardiocare-hero-title">📊 Prediction & Evaluation</div>
        <p class="cardiocare-hero-text">Generate a cardiovascular prediction and evaluate the selected model using the configured evaluation method.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)
    st.header("👤 Patient Prediction")

    st.caption(
        "Enter patient information. Required scaling is "
        "automatically handled by the selected model."
    )

    st.markdown('<div class="section-label">Patient Information</div>', unsafe_allow_html=True)
    st.subheader("Patient Details")

    c1, c2, c3 = st.columns(3)

    with c1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=50
        )

    with c2:

        gender = st.selectbox(
            "Gender",
            [1, 2],
            format_func=lambda x:
            "Female" if x == 1 else "Male"
        )

    with c3:

        height = st.number_input(
            "Height (cm)",
            min_value=50.0,
            max_value=250.0,
            value=170.0,
            step=0.5
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=300.0,
            value=70.0,
            step=0.5
        )

    with c2:

        ap_hi = st.number_input(
            "Systolic BP",
            min_value=50,
            max_value=300,
            value=120
        )

    with c3:

        ap_lo = st.number_input(
            "Diastolic BP",
            min_value=30,
            max_value=200,
            value=80
        )

    st.markdown('<div class="section-label">Medical Information</div>', unsafe_allow_html=True)
    st.subheader("Medical Information")

    c1, c2, c3 = st.columns(3)

    with c1:

        cholesterol = st.selectbox(
            "Cholesterol",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[x]
        )

    with c2:

        gluc = st.selectbox(
            "Glucose",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[x]
        )

    with c3:

        smoke = st.selectbox(
            "Smoking",
            [0, 1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )

    st.markdown('<div class="section-label">Lifestyle Information</div>', unsafe_allow_html=True)
    st.subheader("Lifestyle")

    c1, c2, c3 = st.columns(3)

    with c1:

        alco = st.selectbox(
            "Alcohol",
            [0, 1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )

    with c2:

        active = st.selectbox(
            "Physical Activity",
            [0, 1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )

    with c3:

        st.info(
            "All 11 features are used by the model."
        )

    input_data = pd.DataFrame(
        [[
            age,
            gender,
            height,
            weight,
            ap_hi,
            ap_lo,
            cholesterol,
            gluc,
            smoke,
            alco,
            active
        ]],
        columns=FEATURES
    )

    st.divider()

    predict_button = st.button(
        "🔍 Predict Cardiovascular Disease",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        with st.spinner(
            "Preparing prediction..."
        ):

            final_model = train_final_model(
                selected_model
            )

            result = int(
                final_model.predict(
                    input_data
                )[0]
            )

            probabilities = (
                final_model.predict_proba(
                    input_data
                )[0]
            )

        no_cardio = (
            probabilities[0] * 100
        )

        cardio = (
            probabilities[1] * 100
        )

        st.divider()

        st.header(
            "📊 Prediction Result"
        )

        if result == 1:

            st.error(
                "⚠️ Cardiovascular Disease Predicted"
            )

        else:

            st.success(
                "✅ No Cardiovascular Disease Predicted"
            )

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "No Cardio Disease",
                f"{no_cardio:.2f}%"
            )

        with c2:

            st.metric(
                "Cardio Disease",
                f"{cardio:.2f}%"
            )

        st.write(
            "Cardiovascular Disease Probability"
        )

        st.progress(
            int(
                min(
                    max(cardio, 0),
                    100
                )
            )
        )

        if cardio < 30:

            st.success(
                "Lower predicted probability"
            )

        elif cardio < 60:

            st.warning(
                "Moderate predicted probability"
            )

        else:

            st.error(
                "Higher predicted probability"
            )

        with st.expander(
            "🔎 View Entered Patient Data"
        ):

            st.dataframe(
                input_data,
                use_container_width=True
            )

        st.info(
            "This prediction is for educational/research "
            "purposes and is not a medical diagnosis."
        )




    st.divider()
    st.markdown('<div class="section-label">Evaluation</div>', unsafe_allow_html=True)
    st.markdown(r"""<div class="evaluation-header"><h2 style="margin:0;">📊 Model Evaluation</h2><p style="margin:0.35rem 0 0 0; opacity:0.7;">Evaluate the same selected model using the configured evaluation method.</p></div>""", unsafe_allow_html=True)

    # EVALUATION
    # ============================================================

    if selected_method == "Holdout":

        (
            actual,
            prediction,
            probability
        ) = evaluate_holdout(
            selected_model,
            test_size
        )

        method_description = (
            f"{int((1 - test_size) * 100)}% Training / "
            f"{int(test_size * 100)}% Testing"
        )


    elif selected_method == "Cross-Validation":

        (
            actual,
            prediction,
            probability
        ) = evaluate_cv(
            selected_model,
            folds
        )

        method_description = (
            f"{folds}-Fold Stratified Cross-Validation"
        )


    else:

        (
            bootstrap_scores,
            actual,
            prediction,
            probability
        ) = evaluate_bootstrap(
            selected_model,
            bootstrap_iterations
        )

        method_description = (
            f"{bootstrap_iterations} Bootstrap Iterations"
        )


    # ============================================================
    performance_tab, matrix_tab, roc_tab, report_tab, setup_tab = st.tabs(["📊 Performance", "🔲 Confusion Matrix", "📈 ROC Curve", "📋 Classification Report", "ℹ️ Evaluation Setup"])

    # PERFORMANCE
    # ============================================================

    with performance_tab:

        st.header(
            "📊 Model Performance"
        )

        st.caption(
            method_description
        )

        if selected_method == "Bootstrap":

            if len(bootstrap_scores) == 0:

                st.warning(
                    "No valid bootstrap results were generated."
                )

            else:

                mean_scores = (
                    bootstrap_scores.mean()
                )

                c1, c2, c3, c4, c5 = st.columns(5)

                with c1:

                    st.metric(
                        "Accuracy",
                        f"{mean_scores['Accuracy'] * 100:.2f}%"
                    )

                with c2:

                    st.metric(
                        "Precision",
                        f"{mean_scores['Precision'] * 100:.2f}%"
                    )

                with c3:

                    st.metric(
                        "Recall",
                        f"{mean_scores['Recall'] * 100:.2f}%"
                    )

                with c4:

                    st.metric(
                        "F1 Score",
                        f"{mean_scores['F1 Score'] * 100:.2f}%"
                    )

                with c5:

                    st.metric(
                        "ROC-AUC",
                        f"{mean_scores['ROC-AUC']:.4f}"
                    )

                st.subheader(
                    "Bootstrap Statistics"
                )

                st.dataframe(
                    bootstrap_scores.describe(),
                    use_container_width=True
                )

        else:

            accuracy = accuracy_score(
                actual,
                prediction
            )

            precision = precision_score(
                actual,
                prediction,
                zero_division=0
            )

            recall = recall_score(
                actual,
                prediction,
                zero_division=0
            )

            f1 = f1_score(
                actual,
                prediction,
                zero_division=0
            )

            auc = roc_auc_score(
                actual,
                probability
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:

                st.metric(
                    "Accuracy",
                    f"{accuracy * 100:.2f}%"
                )

            with c2:

                st.metric(
                    "Precision",
                    f"{precision * 100:.2f}%"
                )

            with c3:

                st.metric(
                    "Recall",
                    f"{recall * 100:.2f}%"
                )

            with c4:

                st.metric(
                    "F1 Score",
                    f"{f1 * 100:.2f}%"
                )

            with c5:

                st.metric(
                    "ROC-AUC",
                    f"{auc:.4f}"
                )


    # ============================================================
    # CONFUSION MATRIX
    # ============================================================

    with matrix_tab:

        st.header(
            "🔲 Confusion Matrix"
        )

        cm = confusion_matrix(
            actual,
            prediction,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        matrix_table = pd.DataFrame(
            cm,
            index=[
                "Actual: No Cardio",
                "Actual: Cardio"
            ],
            columns=[
                "Predicted: No Cardio",
                "Predicted: Cardio"
            ]
        )

        st.dataframe(
            matrix_table,
            use_container_width=True
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "True Negative",
                f"{tn:,}"
            )

        with c2:

            st.metric(
                "False Positive",
                f"{fp:,}"
            )

        with c3:

            st.metric(
                "False Negative",
                f"{fn:,}"
            )

        with c4:

            st.metric(
                "True Positive",
                f"{tp:,}"
            )

        st.info(
            "TN = correctly predicted No Cardio | "
            "FP = predicted Cardio but actually No Cardio | "
            "FN = predicted No Cardio but actually Cardio | "
            "TP = correctly predicted Cardio"
        )


    # ============================================================
    # ROC CURVE
    # ============================================================

    with roc_tab:

        st.header(
            "📈 ROC Curve"
        )

        if len(np.unique(actual)) < 2:

            st.warning(
                "ROC curve requires both classes."
            )

        else:

            false_positive_rate, true_positive_rate, _ = (
                roc_curve(
                    actual,
                    probability
                )
            )

            auc = roc_auc_score(
                actual,
                probability
            )

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"{selected_model} (AUC = {auc:.4f})"
            )

            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                label="Random Classifier"
            )

            ax.set_xlabel(
                "False Positive Rate"
            )

            ax.set_ylabel(
                "True Positive Rate"
            )

            ax.set_title(
                f"ROC Curve - {selected_model}"
            )

            ax.legend()

            ax.grid(
                alpha=0.2
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            st.metric(
                "ROC-AUC",
                f"{auc:.4f}"
            )

            plt.close(fig)


    # ============================================================
    # CLASSIFICATION REPORT
    # ============================================================

    with report_tab:

        st.header(
            "📋 Classification Report"
        )

        report = classification_report(
            actual,
            prediction,
            target_names=[
                "No Cardio",
                "Cardio"
            ],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df,
            use_container_width=True
        )

        st.info(
            "Precision measures prediction correctness. "
            "Recall measures how many actual positive cases "
            "were identified. F1 combines precision and recall."
        )


    # ============================================================

    with setup_tab:
        st.subheader("📊 Evaluation Overview")
        st.write(method_description)
        st.metric("Selected Model", selected_model)
        st.metric("Evaluation Method", selected_method)
        st.info("Use the tabs above to inspect detailed performance, confusion matrix, ROC curve and classification report.")



def tuning_page():
    if "tuning_result" not in st.session_state:
        st.session_state.tuning_result = None

    if "tuning_all_result" not in st.session_state:
        st.session_state.tuning_all_result = None


    st.markdown("""
    <style>
    .tuning-header {
        padding: 1.25rem 1.4rem;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        margin-top: 1.4rem;
        margin-bottom: 1rem;
    }
    .tuning-note {
        padding: 0.85rem 1rem;
        border-radius: 10px;
        background: rgba(128,128,128,0.08);
        border-left: 4px solid #6c63ff;
        margin-bottom: 1rem;
    }
    .param-box {
        padding: 0.8rem 1rem;
        border: 1px solid rgba(128,128,128,0.22);
        border-radius: 10px;
        margin-bottom: 0.45rem;
        font-size: 0.92rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """<div class='tuning-header'>
        <h2 style='margin:0;'>⚙️ Hyperparameter Tuning</h2>
        <p style='margin:0.35rem 0 0 0; opacity:0.75;'>
        Optimize model settings with RandomizedSearchCV without changing the existing evaluation workflow.
        </p>
        </div>""",
        unsafe_allow_html=True
    )

    st.markdown(
        """<div class='tuning-note'>
        <b>How it works:</b> 8 parameter combinations × 3-fold stratified cross-validation.
        The best model is then evaluated on a separate test split and saved as a <code>.pkl</code> file.
        </div>""",
        unsafe_allow_html=True
    )


    tune_tab, all_tab = st.tabs([
        "🎯 Tune Selected Model",
        "📊 Tune All Models"
    ])

    with tune_tab:

        info_col, action_col = st.columns([2.2, 1], gap="large")

        with info_col:
            st.markdown(f"### {selected_model}")
            st.write(
                "Tune the currently selected model using the same dataset and "
                "fixed random state used throughout the application."
            )
            st.caption(
                "Existing Holdout, Cross-Validation, Bootstrap, prediction, and comparison features remain unchanged."
            )

        with action_col:
            if st.button(
                "🔧 Start Tuning",
                type="primary",
                use_container_width=True,
                key="tune_selected_main"
            ):
                with st.spinner(f"Tuning {selected_model}... This may take a moment."):
                    tuned_model = tune_model(selected_model)
                st.session_state.tuning_result = tuned_model._cardiocare_tuning_result_
                st.session_state.tuning_result["model_name"] = selected_model
                st.success("Tuning completed and model saved.")

        result = st.session_state.tuning_result

        if result is not None and result.get("model_name") == selected_model:
            st.divider()
            st.markdown("### 📈 Tuning Results")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(
                    "Best CV Accuracy",
                    f"{result['best_cv_accuracy'] * 100:.2f}%"
                )
            with m2:
                st.metric(
                    "Tuned Test Accuracy",
                    f"{result['test_accuracy'] * 100:.2f}%"
                )
            with m3:
                st.metric(
                    "Saved Model",
                    "✓ .pkl"
                )

            params_col, metrics_col = st.columns([1.2, 1], gap="large")

            with params_col:
                st.markdown("#### 🧩 Best Parameters")
                for key, value in result["best_params"].items():
                    st.markdown(
                        f"<div class='param-box'><b>{key}</b><br>{value}</div>",
                        unsafe_allow_html=True
                    )

            with metrics_col:
                st.markdown("#### 📋 Test Set Performance")
                performance_df = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                    "Score": [
                        result["test_accuracy"],
                        result["test_precision"],
                        result["test_recall"],
                        result["test_f1"]
                    ]
                })
                performance_df["Score"] = performance_df["Score"].map(
                    lambda x: f"{x * 100:.2f}%"
                )
                st.dataframe(
                    performance_df,
                    hide_index=True,
                    use_container_width=True
                )

            st.info(
                "The tuned model is stored in the tuned_models folder, so later runs can load the saved .pkl instead of tuning again."
            )

        else:
            st.info(
                f"Ready to tune **{selected_model}**. Click **Start Tuning** to find its best parameter combination."
            )


    with all_tab:

        st.markdown("### 📊 Compare Tuned Models")
        st.write(
            "Tune all five models and view their tuned test performance in one place. "
            "The first run may take longer; saved models are reused on later runs."
        )

        if st.button(
            "🚀 Tune All Five Models",
            type="primary",
            use_container_width=False,
            key="tune_all_main"
        ):
            with st.spinner(
                "Tuning all five models... The first run can take some time."
            ):
                st.session_state.tuning_all_result = tune_all_models()
            st.success("All model tuning completed.")

        if st.session_state.tuning_all_result is not None:
            all_df = st.session_state.tuning_all_result.copy()
            st.divider()
            st.dataframe(
                all_df,
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No all-model tuning results yet. Click the button above to start.")

    st.divider()


def model_info_page():
    st.markdown(r"""
    <div class="cardiocare-hero">
        <div class="cardiocare-hero-title">🤖 Model Information</div>
        <p class="cardiocare-hero-text">Reference information for the five machine-learning models used by CardioCare AI.</p>
    </div>
    """, unsafe_allow_html=True)
    models = {
        "Logistic Regression": "StandardScaler → Logistic Regression",
        "Decision Tree": "Controlled depth, minimum split and minimum leaf size.",
        "KNN": "StandardScaler → KNN with k=25 and distance weighting.",
        "Random Forest": "Random Forest with n_estimators=150, max_depth=10, min_samples_split=5, min_samples_leaf=2 and sqrt features.",
        "AdaBoost": "AdaBoost with 150 estimators, learning_rate=0.5 and a depth-3 Decision Tree base estimator."
    }
    for name, description in models.items():
        with st.expander(name, expanded=(name == selected_model)):
            st.write(description)
    st.divider()
    st.subheader("Input Features")
    st.dataframe(pd.DataFrame({"Feature": FEATURES}), hide_index=True, use_container_width=True)


pages = [
    st.Page(dashboard_page, title="Dashboard", icon="🏠"),
    st.Page(prediction_and_evaluation_page, title="Prediction & Evaluation", icon="📊"),
    st.Page(tuning_page, title="Hyperparameter Tuning", icon="⚙️"),
    st.Page(model_info_page, title="Models", icon="🤖"),
]

pg = st.navigation(pages, position="sidebar")
pg.run()

