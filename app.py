import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

DATA_PATH = "/mnt/data/datasaurus_dino.csv"

st.set_page_config(page_title="Interactive Scatterplot", layout="wide")
st.title("Interactive Scatterplot (Datasaurus Dino)")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic sanity checks / cleanup
    df = df.rename(columns=str.strip)
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError(f"Expected columns x,y but found: {list(df.columns)}")
    df = df.dropna(subset=["x", "y"]).copy()
    return df

df = load_data(DATA_PATH)

with st.sidebar:
    st.header("Controls")

    marker_size = st.slider("Marker size", 3, 20, 8)
    marker_opacity = st.slider("Opacity", 0.05, 1.0, 0.8)
    show_trend = st.checkbox("Show trendline (OLS)", value=False)

    st.subheader("Filters")
    x_min, x_max = float(df["x"].min()), float(df["x"].max())
    y_min, y_max = float(df["y"].min()), float(df["y"].max())
    x_range = st.slider("x range", min_value=x_min, max_value=x_max, value=(x_min, x_max))
    y_range = st.slider("y range", min_value=y_min, max_value=y_max, value=(y_min, y_max))

    st.subheader("Optional jitter")
    jitter_on = st.checkbox("Add jitter (helps reveal overlaps)", value=False)
    jitter_amt = st.slider("Jitter amount", 0.0, 1.5, 0.0, 0.05, disabled=not jitter_on)

plot_df = df[
    (df["x"].between(x_range[0], x_range[1])) &
    (df["y"].between(y_range[0], y_range[1]))
].copy()

if jitter_on and jitter_amt > 0:
    rng = np.random.default_rng(42)
    plot_df["x_plot"] = plot_df["x"] + rng.normal(0, jitter_amt, size=len(plot_df))
    plot_df["y_plot"] = plot_df["y"] + rng.normal(0, jitter_amt, size=len(plot_df))
else:
    plot_df["x_plot"] = plot_df["x"]
    plot_df["y_plot"] = plot_df["y"]

trend = "ols" if show_trend else None

fig = px.scatter(
    plot_df,
    x="x_plot",
    y="y_plot",
    hover_data={"x": True, "y": True, "x_plot": False, "y_plot": False},
    trendline=trend,
    title="Zoom, pan, box-select, and hover",
)

fig.update_traces(marker=dict(size=marker_size, opacity=marker_opacity))
fig.update_layout(
    margin=dict(l=10, r=10, t=60, b=10),
    dragmode="zoom",  # user can switch to select tools in modebar
)

# --- Interactive selection (works on newer Streamlit versions) ---
st.caption("Tip: Use the Plotly modebar (top-right) to box/lasso select points.")

selection = None
try:
    selection = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="scatter",
    )
except TypeError:
    # Older Streamlit: no on_select/selection_mode
    st.plotly_chart(fig, use_container_width=True)

# Summary panels
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (after filters)", f"{len(plot_df):,}")
c2.metric("x mean", f"{plot_df['x'].mean():.3f}")
c3.metric("y mean", f"{plot_df['y'].mean():.3f}")
c4.metric("Corr(x,y)", f"{plot_df[['x','y']].corr().iloc[0,1]:.3f}")

# If selection is available, show selected-point stats
if isinstance(selection, dict) and selection.get("selection"):
    pts = selection["selection"].get("points", [])
    if pts:
        idx = [p.get("pointIndex") for p in pts if p.get("pointIndex") is not None]
        sel = plot_df.iloc[idx].copy()

        st.subheader("Selected points")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Selected count", f"{len(sel):,}")
        sc2.metric("Selected x mean", f"{sel['x'].mean():.3f}")
        sc3.metric("Selected y mean", f"{sel['y'].mean():.3f}")

        st.dataframe(sel[["x", "y"]].reset_index(drop=True), use_container_width=True)
else:
    st.info(
        "Point selection summary appears here on newer Streamlit versions. "
        "If you don’t see it, upgrade Streamlit (e.g., `pip install -U streamlit`)."
    )

with st.expander("Show raw data"):
    st.dataframe(df, use_container_width=True)