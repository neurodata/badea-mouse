import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    # items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax.get_xaxis().get_label(), ax.get_yaxis().get_label()]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)

    # Save just the portion _inside_ the second axis's boundaries
    # extent = full_extent(ax[4]).transformed(fig.dpi_scale_trans.inverted())
    # Alternatively,
    # extent = ax.get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig("ax4_figure.pdf", bbox_inches=extent)
