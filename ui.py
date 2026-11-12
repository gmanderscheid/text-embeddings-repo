"""Shared look of the demo pages: black background, light text, the course palette.

The demos are shown by the trainer on a screen (not worked through by students like the labs), so
the app is dark, in the spirit of the animated explanations it borrows from. The Streamlit side of
the theme is in .streamlit/config.toml; this file sets the plotly side and the page helpers.
"""
from contextlib import contextmanager

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Same palette as the notebooks, plus the dark-theme neutrals
COLORS = ["#FE0162", "#345FFF", "#0ECDD0", "#F6970A", "#58C4DD"]
BG, PANEL, GRID, TEXT, MUTED = "#000000", "#111418", "#22262E", "#ECEFF4", "#8B949E"
THEME_NAME = "dlf_dark"
# sequential scale for heat maps: from the background to blue
BLUES = [[0, "#0B0F17"], [0.25, "#1B2A5C"], [1, "#4C7DFF"]]


def set_plotly_theme():
    "Register the dark plotly template of the demos and make it the default."
    if THEME_NAME not in pio.templates:
        pio.templates[THEME_NAME] = go.layout.Template(layout=dict(
            colorway=COLORS,
            font=dict(color=TEXT, size=13),
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            title=dict(x=0, xanchor="left", font=dict(size=18)),
            xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
            yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
            margin=dict(l=40, r=20, t=60, b=60),
            hoverlabel=dict(bgcolor=PANEL, font_color=TEXT),
        ))
    pio.templates.default = THEME_NAME


def page_header(title, explanation):
    "Title and two lines of explanation, identical on every page."
    st.title(title)
    st.markdown(explanation)


@contextmanager
def sidebar_section(title):
    "A titled block of controls in the sidebar."
    with st.sidebar:
        st.subheader(title)
        yield
        st.divider()


def what_to_look_at(text):
    "A closed expander with questions to ask the room, never the answers."
    with st.expander("👀 What to look at"):
        st.markdown(text)


def figure(height=420):
    "An empty plotly figure with the dark theme, ready to receive traces."
    fig = go.Figure()
    fig.update_layout(height=height, template=THEME_NAME)
    return fig


def embed_animation(name, data=None, height=800):
    "Embed components/<name>.html, with `data` injected as JSON in place of __DATA__."
    import json
    from pathlib import Path
    import streamlit.components.v1 as components
    html = (Path(__file__).resolve().parent / "components" / f"{name}.html").read_text()
    if data is not None:
        html = html.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    if hasattr(st, "iframe"):                       # Streamlit >= 1.52
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


def plotly_kwargs():
    "Keep the dark template of this file (theme=None), on old and new Streamlit."
    import inspect
    return {"theme": None} if "width" in inspect.signature(st.plotly_chart).parameters \
        else {"theme": None, "use_container_width": True}
