# ============================================================
# STREAMLIT APP
# Surface Roughness Prediction in CNC Turning
# Direct + Inverse Prediction using MLP
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Surface Roughness Prediction",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Surface Roughness Prediction in CNC Turning")
st.write("Direct and inverse prediction using a neural network model (MLP).")


# ============================================================
# DATA AND MODEL
# ============================================================

@st.cache_data
def load_data():
    data = [
        (1,1,1,1,1,0.814),(2,1,1,2,2,3.184),(3,1,1,3,3,5.712),
        (4,1,2,1,1,1.023),(5,1,2,2,2,3.460),(6,1,2,3,3,5.685),
        (7,1,3,1,2,0.960),(8,1,3,2,3,3.204),(9,1,3,3,1,5.870),
        (10,2,1,1,3,1.681),(11,2,1,2,1,3.474),(12,2,1,3,2,6.281),
        (13,2,2,1,2,1.737),(14,2,2,2,3,3.586),(15,2,2,3,1,5.988),
        (16,2,3,1,3,1.140),(17,2,3,2,1,3.073),(18,2,3,3,2,5.606),
        (19,1,1,1,1,1.164),(20,1,1,2,2,3.053),(21,1,1,3,3,9.448),
        (22,1,2,1,1,1.173),(23,1,2,2,2,3.382),(24,1,2,3,3,6.146),
        (25,1,3,1,2,1.081),(26,1,3,2,3,3.461),(27,1,3,3,1,5.540),
        (28,2,1,1,3,1.661),(29,2,1,2,1,3.158),(30,2,1,3,2,6.277),
        (31,2,2,1,2,1.776),(32,2,2,2,3,3.138),(33,2,2,3,1,5.171),
        (34,2,3,1,3,1.070),(35,2,3,2,1,3.072),(36,2,3,3,2,5.494),
        (37,1,1,1,1,0.993),(38,1,1,2,2,2.993),(39,1,1,3,3,6.251),
        (40,1,2,1,1,1.085),(41,1,2,2,2,1.062),(42,1,2,3,3,5.624),
        (43,1,3,1,2,1.413),(44,1,3,2,3,3.560),(45,1,3,3,1,5.592),
        (46,2,1,1,3,1.832),(47,2,1,2,1,3.323),(48,2,1,3,2,5.527),
        (49,2,2,1,2,1.612),(50,2,2,2,3,3.349),(51,2,2,3,1,5.125),
        (52,2,3,1,3,1.189),(53,2,3,2,1,2.691),(54,2,3,3,2,5.224),

        # External validation
        (55,1,3,1,1,0.766),
        (56,1,1,2,3,2.569),
        (57,1,2.5,2.5,1.5,3.408),
        (58,1,1.5,2.5,2.5,4.046),
        (59,1,2,2,1,3.467),
        (60,2,3,1,1,1.055),
        (61,2,2,1,3,1.268),
        (62,2,3,3,3,5.724),
        (63,2,1.5,1.5,1.5,1.944),
        (64,2,2.5,1.5,2.5,1.592),
    ]

    df = pd.DataFrame(
        data,
        columns=["Run", "Insert", "Vc_level", "Feed_level", "ap_level", "Ra"]
    )

    vc_map = {1: 200, 2: 250, 3: 300}
    feed_map = {1: 0.15, 2: 0.25, 3: 0.35}
    ap_map = {1: 0.5, 2: 1.0, 3: 2.0}

    def coded_to_real(value, mapping):
        levels = np.array(list(mapping.keys()), dtype=float)
        real_values = np.array(list(mapping.values()), dtype=float)
        return np.interp(value, levels, real_values)

    df["Vc"] = df["Vc_level"].apply(lambda x: coded_to_real(x, vc_map))
    df["Feed"] = df["Feed_level"].apply(lambda x: coded_to_real(x, feed_map))
    df["ap"] = df["ap_level"].apply(lambda x: coded_to_real(x, ap_map))

    return df


@st.cache_resource
def train_model(df):
    train_df = df[df["Run"] <= 54].copy()
    val_df = df[df["Run"] >= 55].copy()

    X_train = train_df[["Insert", "Vc", "Feed", "ap"]]
    y_train = train_df["Ra"]

    X_val = val_df[["Insert", "Vc", "Feed", "ap"]]
    y_val = val_df["Ra"]

    preprocess = ColumnTransformer(
        transformers=[
            ("scale", MinMaxScaler(), ["Vc", "Feed", "ap"]),
            ("insert", "passthrough", ["Insert"])
        ]
    )

    mlp_model = MLPRegressor(
        hidden_layer_sizes=(10, 10),
        activation="relu",
        solver="adam",
        max_iter=5000,
        random_state=42
    )

    model = Pipeline([
        ("preprocess", preprocess),
        ("mlp", mlp_model)
    ])

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "val_rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
        "val_mae": mean_absolute_error(y_val, y_val_pred),
        "val_r2": r2_score(y_val, y_val_pred),
    }

    validation_results = val_df[["Run", "Insert", "Vc", "Feed", "ap", "Ra"]].copy()
    validation_results["Predicted_Ra"] = y_val_pred
    validation_results["Absolute_Error"] = abs(
        validation_results["Ra"] - validation_results["Predicted_Ra"]
    )
    validation_results["Percentage_Error"] = (
        100 * validation_results["Absolute_Error"] / validation_results["Ra"]
    )

    return model, train_df, val_df, X_train, y_train, X_val, y_val, y_val_pred, metrics, validation_results


df = load_data()
model, train_df, val_df, X_train, y_train, X_val, y_val, y_val_pred, metrics, validation_results = train_model(df)


# ============================================================
# FUNCTIONS
# ============================================================

def predict_ra(insert, vc, feed, ap):
    input_data = pd.DataFrame({
        "Insert": [insert],
        "Vc": [vc],
        "Feed": [feed],
        "ap": [ap]
    })
    return model.predict(input_data)[0]


def inverse_prediction(ra_target):
    best_solution = None

    for insert in [1, 2]:

        def objective(params):
            vc, feed, ap = params
            ra_pred = predict_ra(insert, vc, feed, ap)
            return abs(ra_target - ra_pred)

        bounds = [
            (200, 300),     # Vc
            (0.15, 0.35),   # Feed
            (0.5, 2.0)      # ap
        ]

        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=250,
            popsize=10,
            tol=1e-6,
            seed=42,
            polish=True
        )

        vc_opt, feed_opt, ap_opt = result.x
        ra_pred_opt = predict_ra(insert, vc_opt, feed_opt, ap_opt)
        error = abs(ra_target - ra_pred_opt)

        solution = {
            "Insert": insert,
            "Vc": vc_opt,
            "Feed": feed_opt,
            "ap": ap_opt,
            "Target_Ra": ra_target,
            "Predicted_Ra": ra_pred_opt,
            "Absolute_Error": error
        }

        if best_solution is None or error < best_solution["Absolute_Error"]:
            best_solution = solution

    return best_solution


def accuracy_message(error):
    if error <= 0.20:
        st.success("High agreement between target and predicted Ra.")
    elif error <= 0.50:
        st.warning("Moderate agreement. The target may be close to a model limit.")
    else:
        st.error("Low agreement. The target may be outside the experimental domain.")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model information")
st.sidebar.write("**Model:** Multilayer Perceptron (MLP)")
st.sidebar.write("**Architecture:** 4 → 10 → 10 → 1")
st.sidebar.write("**Training data:** 54 samples")
st.sidebar.write("**External validation:** 10 samples")

st.sidebar.markdown("---")
st.sidebar.write("**Experimental domain**")
st.sidebar.write("Vc: 200–300 m/min")
st.sidebar.write("Feed: 0.15–0.35 mm/rev")
st.sidebar.write("ap: 0.5–2.0 mm")
st.sidebar.write("Insert: 1 or 2")


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Validation RMSE", f"{metrics['val_rmse']:.4f} µm")

with col2:
    st.metric("Validation MAE", f"{metrics['val_mae']:.4f} µm")

with col3:
    st.metric("Validation R²", f"{metrics['val_r2']:.4f}")


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Direct Prediction",
    "Inverse Prediction",
    "Model Validation",
    "Response Surface",
    "Dataset"
])


# ============================================================
# TAB 1 - DIRECT PREDICTION
# ============================================================

with tab1:
    st.subheader("Direct Prediction")
    st.write("Enter machining parameters to predict surface roughness.")

    with st.form("direct_form"):
        c1, c2 = st.columns(2)

        with c1:
            insert = st.selectbox("Insert type", [1, 2])
            vc = st.slider("Cutting speed Vc (m/min)", 200.0, 300.0, 250.0, step=1.0)

        with c2:
            feed = st.slider("Feed rate f (mm/rev)", 0.15, 0.35, 0.25, step=0.005)
            ap = st.slider("Depth of cut ap (mm)", 0.5, 2.0, 1.0, step=0.05)

        submitted_direct = st.form_submit_button("Predict Ra")

    if submitted_direct:
        ra = predict_ra(insert, vc, feed, ap)

        st.success(f"Predicted surface roughness: **{ra:.4f} µm**")

        result_df = pd.DataFrame({
            "Insert": [insert],
            "Vc (m/min)": [vc],
            "Feed (mm/rev)": [feed],
            "ap (mm)": [ap],
            "Predicted Ra (µm)": [ra]
        })

        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download result as CSV",
            data=csv,
            file_name="direct_prediction_result.csv",
            mime="text/csv"
        )


# ============================================================
# TAB 2 - INVERSE PREDICTION
# ============================================================

with tab2:
    st.subheader("Inverse Prediction")
    st.write("Enter a target surface roughness and the model will recommend machining parameters.")

    with st.form("inverse_form"):
        ra_target = st.number_input(
            "Target Ra (µm)",
            min_value=0.10,
            max_value=10.00,
            value=1.50,
            step=0.01
        )

        submitted_inverse = st.form_submit_button("Find Parameters")

    if submitted_inverse:
        with st.spinner("Optimizing machining parameters..."):
            sol = inverse_prediction(ra_target)

        st.write("### Prediction result")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Target Ra", f"{sol['Target_Ra']:.4f} µm")

        with c2:
            st.metric("Predicted Ra", f"{sol['Predicted_Ra']:.4f} µm")

        with c3:
            st.metric("Absolute error", f"{sol['Absolute_Error']:.4f} µm")

        accuracy_message(sol["Absolute_Error"])

        st.write("### Recommended machining parameters")

        inverse_df = pd.DataFrame({
            "Insert": [sol["Insert"]],
            "Vc (m/min)": [sol["Vc"]],
            "Feed (mm/rev)": [sol["Feed"]],
            "ap (mm)": [sol["ap"]],
            "Target Ra (µm)": [sol["Target_Ra"]],
            "Predicted Ra (µm)": [sol["Predicted_Ra"]],
            "Absolute error (µm)": [sol["Absolute_Error"]]
        })

        st.dataframe(inverse_df, use_container_width=True)

        csv = inverse_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download inverse result as CSV",
            data=csv,
            file_name="inverse_prediction_result.csv",
            mime="text/csv"
        )

        st.info(
            "This result is a model-based recommendation. "
            "Experimental validation is required before using it as a final machining condition."
        )


# ============================================================
# TAB 3 - MODEL VALIDATION
# ============================================================

with tab3:
    st.subheader("Model Validation")

    st.write("### External validation results")
    st.dataframe(validation_results, use_container_width=True)

    csv = validation_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download validation table as CSV",
        data=csv,
        file_name="external_validation_results.csv",
        mime="text/csv"
    )

    st.write("### Experimental vs predicted Ra")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_val, y_val_pred)
    min_val = min(y_val.min(), y_val_pred.min())
    max_val = max(y_val.max(), y_val_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Ideal prediction")
    ax.set_xlabel("Experimental Ra (µm)")
    ax.set_ylabel("Predicted Ra (µm)")
    ax.set_title("External Validation: Experimental vs Predicted Ra")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig)

    st.write("### Prediction error distribution")

    errors = y_val.values - y_val_pred

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(errors, bins=8)
    ax2.axvline(0, linestyle="--")
    ax2.set_xlabel("Prediction error (Experimental - Predicted) [µm]")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Prediction Error Distribution")
    ax2.grid(True, alpha=0.25)
    st.pyplot(fig2)

    st.write("### Feature importance")

    try:
        perm_result = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=30,
            random_state=42,
            scoring="neg_root_mean_squared_error"
        )

        importance_df = pd.DataFrame({
            "Feature": ["Insert", "Cutting speed Vc", "Feed rate f", "Depth of cut ap"],
            "Importance": perm_result.importances_mean
        }).sort_values("Importance", ascending=False)

        st.dataframe(importance_df, use_container_width=True)

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.bar(importance_df["Feature"], importance_df["Importance"])
        ax3.set_ylabel("Mean increase in RMSE (µm)")
        ax3.set_xlabel("Input variable")
        ax3.set_title("Permutation Feature Importance")
        ax3.tick_params(axis="x", rotation=20)
        ax3.grid(axis="y", alpha=0.25)
        st.pyplot(fig3)

    except Exception as e:
        st.warning(f"Feature importance could not be calculated: {e}")


# ============================================================
# TAB 4 - RESPONSE SURFACE
# ============================================================

with tab4:
    st.subheader("Response Surface")
    st.write("Visualize the predicted roughness over cutting speed and feed rate.")

    c1, c2 = st.columns(2)

    with c1:
        insert_surface = st.selectbox("Insert type for surface", [1, 2], key="surface_insert")

    with c2:
        ap_surface = st.slider(
            "Fixed depth of cut ap (mm)",
            0.5,
            2.0,
            1.0,
            step=0.05,
            key="surface_ap"
        )

    vc_range = np.linspace(200, 300, 40)
    feed_range = np.linspace(0.15, 0.35, 40)

    Z = np.zeros((len(vc_range), len(feed_range)))

    for i, vc_value in enumerate(vc_range):
        for j, feed_value in enumerate(feed_range):
            Z[i, j] = predict_ra(insert_surface, vc_value, feed_value, ap_surface)

    fig4, ax4 = plt.subplots(figsize=(7, 5))
    contour = ax4.contourf(feed_range, vc_range, Z, levels=20)
    fig4.colorbar(contour, ax=ax4, label="Predicted Ra (µm)")
    ax4.set_xlabel("Feed rate f (mm/rev)")
    ax4.set_ylabel("Cutting speed Vc (m/min)")
    ax4.set_title(f"Response Surface: Insert {insert_surface}, ap = {ap_surface:.2f} mm")
    st.pyplot(fig4)


# ============================================================
# TAB 5 - DATASET
# ============================================================

with tab5:
    st.subheader("Experimental Dataset")
    st.write("The first 54 samples are used for training. Samples 55–64 are used for external validation.")

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download complete dataset as CSV",
        data=csv,
        file_name="surface_roughness_dataset.csv",
        mime="text/csv"
    )

    st.write("### Training and validation split")

    split_df = pd.DataFrame({
        "Subset": ["Training", "External validation"],
        "Number of samples": [len(train_df), len(val_df)]
    })

    st.dataframe(split_df, use_container_width=True)