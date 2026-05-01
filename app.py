# ============================================================
# STREAMLIT APP
# Surface Roughness Prediction (Direct + Inverse)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

from scipy.optimize import differential_evolution


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(page_title="Surface Roughness App", layout="wide")

st.title("🔬 Surface Roughness Prediction in CNC Turning")
st.write("Direct and inverse prediction using a neural network (MLP)")


# =========================
# DATASET
# =========================

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

    # validation
    (55,1,3,1,1,0.766),(56,1,1,2,3,2.569),
    (57,1,2.5,2.5,1.5,3.408),(58,1,1.5,2.5,2.5,4.046),
    (59,1,2,2,1,3.467),(60,2,3,1,1,1.055),
    (61,2,2,1,3,1.268),(62,2,3,3,3,5.724),
    (63,2,1.5,1.5,1.5,1.944),(64,2,2.5,1.5,2.5,1.592),
]

df = pd.DataFrame(data, columns=["Run","Insert","Vc_level","Feed_level","ap_level","Ra"])


# =========================
# CONVERSION
# =========================

vc_map = {1:200,2:250,3:300}
feed_map = {1:0.15,2:0.25,3:0.35}
ap_map = {1:0.5,2:1.0,3:2.0}

def coded_to_real(value, mapping):
    levels = np.array(list(mapping.keys()), dtype=float)
    real_values = np.array(list(mapping.values()), dtype=float)
    return np.interp(value, levels, real_values)

df["Vc"] = df["Vc_level"].apply(lambda x: coded_to_real(x, vc_map))
df["Feed"] = df["Feed_level"].apply(lambda x: coded_to_real(x, feed_map))
df["ap"] = df["ap_level"].apply(lambda x: coded_to_real(x, ap_map))


# =========================
# TRAIN MODEL
# =========================

train_df = df[df["Run"] <= 54]

X = train_df[["Insert","Vc","Feed","ap"]]
y = train_df["Ra"]

preprocess = ColumnTransformer([
    ("scale", MinMaxScaler(), ["Vc","Feed","ap"]),
    ("insert","passthrough",["Insert"])
])

model = Pipeline([
    ("preprocess", preprocess),
    ("mlp", MLPRegressor(hidden_layer_sizes=(10,10), max_iter=5000, random_state=42))
])

model.fit(X,y)


# =========================
# FUNCTIONS
# =========================

def predict_ra(insert, vc, feed, ap):
    df_input = pd.DataFrame({
        "Insert":[insert],
        "Vc":[vc],
        "Feed":[feed],
        "ap":[ap]
    })
    return model.predict(df_input)[0]


def inverse_prediction(ra_target):
    best = None

    for insert in [1,2]:

        def objective(params):
            vc, feed, ap = params
            return abs(ra_target - predict_ra(insert, vc, feed, ap))

        bounds = [(200,300),(0.15,0.35),(0.5,2.0)]

        result = differential_evolution(objective, bounds, seed=42)

        vc, feed, ap = result.x
        ra_pred = predict_ra(insert, vc, feed, ap)

        sol = {
            "Insert": insert,
            "Vc": vc,
            "Feed": feed,
            "ap": ap,
            "Predicted_Ra": ra_pred,
            "Error": abs(ra_target - ra_pred)
        }

        if best is None or sol["Error"] < best["Error"]:
            best = sol

    return best


# =========================
# TABS
# =========================

tab1, tab2 = st.tabs(["Direct Prediction","Inverse Prediction"])


# =========================
# DIRECT
# =========================

with tab1:
    st.subheader("Direct Prediction")

    insert = st.selectbox("Insert type",[1,2])
    vc = st.slider("Vc (m/min)",200.0,300.0,250.0)
    feed = st.slider("Feed (mm/rev)",0.15,0.35,0.25)
    ap = st.slider("ap (mm)",0.5,2.0,1.0)

    if st.button("Predict Ra"):
        ra = predict_ra(insert, vc, feed, ap)
        st.success(f"Predicted Ra: {ra:.4f} µm")


# =========================
# INVERSE
# =========================

with tab2:
    st.subheader("Inverse Prediction")

    with st.form("inverse_form"):

        ra_target = st.number_input(
            "Target Ra (µm)",
            min_value=0.5,
            max_value=10.0,
            value=1.5
        )

        submitted = st.form_submit_button("Find Parameters")

    if submitted:
        with st.spinner("Optimizing..."):
            sol = inverse_prediction(ra_target)

        st.success("Solution found!")

        st.write("### Result")
        st.write(f"Predicted Ra: {sol['Predicted_Ra']:.4f} µm")
        st.write(f"Error: {sol['Error']:.4f} µm")

        st.write("### Recommended parameters")
        st.write(f"Insert: {sol['Insert']}")
        st.write(f"Vc: {sol['Vc']:.2f}")
        st.write(f"Feed: {sol['Feed']:.4f}")
        st.write(f"ap: {sol['ap']:.4f}")

        if sol["Error"] > 0.25:
            st.warning("Target may be outside experimental domain")