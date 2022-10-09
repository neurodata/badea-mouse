import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def set_theme(
    theme=None,
    spine_right=False,
    spine_top=False,
    spine_left=True,
    spine_bottom=True,
    axes_edgecolor="black",
    tick_color="black",
    axes_labelcolor="black",
    text_color="black",
    context="talk",
    tick_size=0,
    font_scale=1,
):
    if theme is None:
        rc_dict = {
            "axes.spines.right": spine_right,
            "axes.spines.top": spine_top,
            "axes.spines.left": spine_left,
            "axes.spines.bottom": spine_bottom,
            "axes.edgecolor": axes_edgecolor,
            "ytick.color": tick_color,
            "xtick.color": tick_color,
            "axes.labelcolor": axes_labelcolor,
            "text.color": text_color,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "xtick.major.size": tick_size,
            "ytick.major.size": tick_size,
        }
