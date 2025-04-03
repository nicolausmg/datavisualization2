import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import io
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, mean_squared_error, confusion_matrix, roc_curve, auc, precision_score, recall_score)
from sklearn.datasets import (load_iris, load_digits, load_diabetes,load_wine, load_breast_cancer)

# ----------------------- Constants and Config ----------------------- #
@st.cache_data
def load_data ():
    dataset_loaders = {
        "Iris": load_iris,
        "Digits": load_digits,
        "Diabetes": load_diabetes,
        "Wine": load_wine,
        "Breast Cancer": load_breast_cancer
    }
    return dataset_loaders

dataset_loaders = load_data()

model_options = {"classification": ["Random Forest", "Logistic Regression"], "regression": ["Random Forest", "Linear Regression"]}

# ----------------------- Streamlit Setup ----------------------- #
st.title("Individual Assignment: Train your own Machine Learning Model")
st.sidebar.title("Data Options")
uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv"])

# ----------------------- Data Loading ----------------------- #
df = None
y = None
target_names = None
is_classification = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    dataset_name = uploaded_file.name
    target_column = st.selectbox("Select the target column", df.columns, index=len(df.columns)-1)
    y = df[target_column]
    df = df.drop(columns=[target_column])
    task_type_choice = st.radio("Select Task Type", ["Classification", "Regression"])
    is_classification = (task_type_choice == "Classification")
    target_names = np.unique(y)

else:
    def on_dataset_change():
        for key in ['trained_model', 'model_choice', 'selected_features', 'is_classification', 'y_test', 'y_pred', 'X_test']:
            st.session_state.pop(key, None)

    dataset_name = st.sidebar.selectbox("Or select a built-in dataset", ["-- Select a Dataset --"] + list(dataset_loaders.keys()), key="dataset_selector", on_change=on_dataset_change)
    if dataset_name == "-- Select a Dataset --":
        st.info("Please upload a CSV or choose a built-in dataset.")
        st.stop()
    data = dataset_loaders[dataset_name]()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    is_classification = len(np.unique(y)) < 20 and np.issubdtype(y.dtype, np.integer)
    target_names = data.target_names if is_classification else np.unique(y)
    st.info("As I mentioned to you in class, the built in datasets don't have categorical varaibles (since I could just access the sklearn datasets, and not those of seaborn). However, the app separates categorical from numerical features if you upload a dataset with those characteristics.")

# ----------------------- Data Preview ----------------------- #
st.write(f"Dataset preview: **{dataset_name}**")
st.dataframe(df, use_container_width=True)
st.write(f"Shape: `{df.shape}`")
st.write("Target Classes:", target_names)

# ----------------------- Feature Selection ----------------------- #
st.write("### Feature Selection")
qualitative_features = df.select_dtypes(include=["object"]).columns.tolist()
quantitative_features = df.select_dtypes(include=["number"]).columns.tolist()

selected_qualitative = st.multiselect("Qualitative Features", qualitative_features, default=qualitative_features)
selected_quantitative = st.multiselect("Quantitative Features", quantitative_features, default=quantitative_features)
selected_features = selected_qualitative + selected_quantitative

if not selected_features:
    st.warning("Please select at least one feature.")
    st.stop()

X = df[selected_features]
target = y
task_type = "classification" if is_classification else "regression"

# ----------------------- Model Configuration ----------------------- #
model_choice = st.sidebar.selectbox("Select Model", model_options[task_type])

with st.sidebar.form("model_config_form"):
    st.header("Model Configuration")

    test_ratio = st.slider("Test Set Size (%)", 10, 50, 20)
    test_size = test_ratio / 100.0

    if model_choice == "Random Forest":
        n_estimators = st.slider("Number of Estimators", 10, 500, 100)
        max_depth = st.slider("Max Depth", 1, 50, 10)
        min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)
        max_features = st.selectbox("Max Features", ["sqrt", "log2", None])
    elif model_choice == "Logistic Regression":
        max_iter = st.slider("Max Iterations", 100, 2000, 1000)
        C = st.slider("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0)

    submitted = st.form_submit_button("Fit Model")

# ----------------------- Model Training ----------------------- #
if submitted:
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=test_ratio, random_state=42)

    if model_choice == "Random Forest":
        model_cls = RandomForestClassifier if is_classification else RandomForestRegressor
        model = model_cls(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=42
        )
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=max_iter, C=C)
    elif model_choice == "Linear Regression":
        model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.session_state.update({
        'trained_model': model,
        'model_choice': model_choice,
        'selected_features': selected_features,
        'is_classification': is_classification,
        'y_test': y_test,
        'y_pred': y_pred,
        'X_test': X_test
    })

# ----------------------- Model Results & Evaluation ----------------------- #
if 'trained_model' in st.session_state:
    model = st.session_state['trained_model']
    y_test = st.session_state['y_test']
    y_pred = st.session_state['y_pred']
    model_choice = st.session_state['model_choice']
    is_classification = st.session_state['is_classification']
    selected_features = st.session_state['selected_features']
    X_test = st.session_state['X_test']

    st.subheader(f"Model Results: {model_choice}")
    if is_classification:
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        st.success(f"Accuracy: {acc:.2f}")
        st.success(f"Precision: {prec:.2f}")
        st.success(f"Recall: {rec:.2f}")
    else:
        mse = mean_squared_error(y_test, y_pred)
        st.success(f"Mean Squared Error: **{mse:.2f}**")

    # ----------------------- Model Download ----------------------- #
    st.subheader("Download Trained Model")
    st.info("As mentioned in class, we never saw how to export a model, so I had to search for it in online documentation. I found that the `joblib` library is used to save and load models in Python. It is a part of the `sklearn` library, which is the one we have used at Esade for our machine learning tasks. The `joblib.dump()` function is used to save the model to a file, and `joblib.load()` is used to load the model back into memory.")
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    st.download_button("Download Model", data=buffer, file_name="trained_model.pkl", mime="application/octet-stream")

    # ----------------------- Visualizations ----------------------- #
    st.subheader("Visualizations")

    if model_choice == "Random Forest":
        st.write("Feature Importances")
        importance_df = pd.DataFrame({"Feature": selected_features, "Importance": model.feature_importances_})
        importance_df = importance_df.sort_values("Importance", ascending=False)
        fig_imp, ax_imp = plt.subplots()
        sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax_imp)
        st.pyplot(fig_imp)

    if is_classification:
        st.write("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        st.pyplot(fig_cm)

        if len(np.unique(y_test)) == 2:
            st.write("#### ROC Curve")
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax_roc.plot([0, 1], [0, 1], linestyle="--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend()
            st.pyplot(fig_roc)
        else:
            st.info("As far as we have seen in previous courses, ROC Curve is not applicable for multi-class classification. Select a binary classification problem to see the ROC Curve (ex. Breast Cancer).")
    else:
        st.write("#### Residuals Distribution")
        residuals = y_test - y_pred
        fig_resid, ax_resid = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax_resid)
        ax_resid.set_title("Residuals Distribution")
        st.pyplot(fig_resid)
