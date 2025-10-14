#!/usr/bin/python3

import tkinter as tk
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pygame

# --- Dynamic 2D Figure ---
class Dynamic2DFigure():
    def __init__(self, 
                 figsize=(8,8), 
                 edgecolor="black", 
                 rect=[0.1, 0.1, 0.8, 0.8],
                 *args, **kwargs):
        self.graphs = {}
        self.texts = {}
        self.fig = plt.Figure(figsize=figsize, edgecolor=edgecolor)
        self.ax = self.fig.add_axes(rect)
        self.fig.tight_layout()
        self.marker_text_offset = 0
        if kwargs.get("title") is not None:
            self.fig.suptitle(kwargs["title"])
        self.axis_equal = False
        self.invert_xaxis = False

    def set_invert_x_axis(self):
        self.invert_xaxis = True

    def set_axis_equal(self):
        self.axis_equal = True

    def add_graph(self, name, label="", window_size=10, x0=None, y0=None,
                  linestyle='-', linewidth=1, marker="", color="k", 
                  markertext=None, marker_text_offset=2):
        self.marker_text_offset = marker_text_offset

        if x0 is None or y0 is None:
            x0 = np.zeros(window_size)
            y0 = np.zeros(window_size)
            new_graph, = self.ax.plot(x0, y0, label=label, 
                                      linestyle=linestyle, linewidth=linewidth,
                                      marker=marker, color=color)
            if markertext is not None:
                new_text = self.ax.text(x0[-1], y0[-1] + marker_text_offset, 
                                         markertext)
        else:
            new_graph, = self.ax.plot(x0, y0, label=label, 
                                      linestyle=linestyle, linewidth=linewidth,
                                      marker=marker, color=color)
            if markertext is not None:
                new_text = self.ax.text(x0[-1], y0[-1] + marker_text_offset, 
                                         markertext)

        self.graphs[name] = new_graph
        if markertext is not None:
            self.texts[name + "_TEXT"] = new_text

    def roll(self, name, new_x, new_y):
        graph = self.graphs[name]
        if graph is not None:
            x, y = graph.get_data()
            x = np.roll(x, -1)
            x[-1] = new_x
            y = np.roll(y, -1)
            y[-1] = new_y
            graph.set_data((x, y))
            self.rescale()
        if name + "_TEXT" in self.texts:
            graph_text = self.texts[name + "_TEXT"]
            x = new_x
            y = new_y + self.marker_text_offset
            graph_text.set_position((x, y))
            self.rescale()

    def update(self, name, new_x_vec, new_y_vec, new_colour='k'):
        graph = self.graphs[name]
        if graph is not None:
            graph.set_data((np.array(new_x_vec), np.array(new_y_vec)))
            graph.set_color(new_colour)
            self.rescale()
        if name + "_TEXT" in self.texts:
            graph_text = self.texts[name + "_TEXT"]
            x = new_x_vec[-1]
            y = new_y_vec[-1] + self.marker_text_offset
            graph_text.set_position((x, y))
            self.rescale()

    def rescale(self):
        xmin = float("inf")
        xmax = -float("inf")
        ymin, ymax = self.ax.get_ylim()
        for name, graph in self.graphs.items():
            xvals, yvals = graph.get_data()
            xmin_data = xvals.min()
            xmax_data = xvals.max()
            ymin_data = yvals.min()
            ymax_data = yvals.max()
            xmin_padded = xmin_data-0.05*(xmax_data-xmin_data)
            xmax_padded = xmax_data+0.05*(xmax_data-xmin_data)
            ymin_padded = ymin_data-0.05*(ymax_data-ymin_data)
            ymax_padded = ymax_data+0.05*(ymax_data-ymin_data)
            xmin = min(xmin_padded, xmin)
            xmax = max(xmax_padded, xmax)
            ymin = min(ymin_padded, ymin)
            ymax = max(ymax_padded, ymax)
        if xmin == float("inf") or xmax == -float("inf"):
            xmin, xmax = 0, 1
        if ymin == float("inf") or ymax == -float("inf"):
            ymin, ymax = 0, 1
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        if self.axis_equal:
            self.ax.set_aspect('equal')
        if self.invert_xaxis:
            self.ax.invert_xaxis()


# --- Dynamic Figure ---
class DynamicFigure():
    def __init__(self, figsize=(5,3), title=None):
        self.graphs = {}
        self.fig = plt.Figure(figsize=(7.3,4.5), edgecolor="black")
        self.ax = self.fig.add_axes([0.12, 0.12, 0.86, 0.86])
        self.fig.tight_layout()
        if title is not None:
            self.fig.suptitle(title)

    def add_graph(self, name, label="", window_size=15, x0=None, y0=None):
        if y0 is None:
            x0 = np.zeros(window_size)
            y0 = np.zeros(window_size)
            new_graph, = self.ax.plot(x0, y0, label=label)
        elif x0 is None:
            new_graph, = self.ax.plot(y0, label=label)
        else:
            new_graph, = self.ax.plot(x0, y0, label=label)
        self.graphs[name] = new_graph

    def roll(self, name, new_x, new_y):
        graph = self.graphs[name]
        if graph is not None:
            x, y = graph.get_data()
            x = np.roll(x, -1)
            x[-1] = new_x
            y = np.roll(y, -1)
            y[-1] = new_y
            graph.set_data((x, y))
            self.rescale()

    def rescale(self):
        xmin = float("inf")
        xmax = -float("inf")
        ymin, ymax = self.ax.get_ylim()
        for name, graph in self.graphs.items():
            xvals, yvals = graph.get_data()
            xmin_data = xvals.min()
            xmax_data = xvals.max()
            ymin_data = yvals.min()
            ymax_data = yvals.max()
            xmin_padded = xmin_data-0.05*(xmax_data-xmin_data)
            xmax_padded = xmax_data+0.05*(xmax_data-xmin_data)
            ymin_padded = ymin_data-0.05*(ymax_data-ymin_data)
            ymax_padded = ymax_data+0.05*(ymax_data-ymin_data)
            xmin = min(xmin_padded, xmin)
            xmax = max(xmax_padded, xmax)
            ymin = min(ymin_padded, ymin)
            ymax = max(ymax_padded, ymax)
        if xmin == float("inf") or xmax == -float("inf"):
            xmin, xmax = 0, 1
        if ymin == float("inf") or ymax == -float("inf"):
            ymin, ymax = 0, 1
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)


# --- Live Plotter ---
class LivePlotter():
    def __init__(self, tk_title=None):
        self._default_w = 150
        self._default_h = 100
        self._graph_w = 0
        self._graph_h = 0
        self._surf_w = 0
        self._surf_h = 0

        self._figs = []
        self._fcas = {}
        self._photos = {}
        self._fig_ids = {}
        self._fig_sizes = {}

        self._text_id = None
        self._empty = True

        self._root = tk.Tk()
        if tk_title is not None:
            self._root.title(tk_title)

        self._screen_w = self._root.winfo_screenwidth()
        self._screen_h = self._root.winfo_screenheight()
        
        # Calculate sim window size as 25% of screen height (square aspect ratio)
        self._sim_window_size = int(self._screen_h * 0.25)

        # Increased default window size for better visibility
        self._window_w = 1400
        self._window_h = 750
        self._root.geometry(f"{self._window_w}x{self._window_h}+0+{self._screen_h - self._window_h}")

        self._canvas = tk.Canvas(self._root, width=self._default_w, height=self._default_h)
        self._canvas.config(bg="#6A6A6A")
        self._text_id = self._canvas.create_text(
            (self._default_w/2, self._default_h/2),
            text="No live plots\ncreated yet.")
        self._canvas.grid(row=0, column=0, sticky="nw")

        self._display = None
        self._game_frame = None
        self._pygame_init = False

        self._surfs = []
        self._surf_coords = {}

    def _rebuild_canvas_layout(self):
        n = len(self._figs)
        if n == 0:
            return

        widths = [self._fig_sizes.get(fig, (0,0))[0] for fig in self._figs]
        heights = [self._fig_sizes.get(fig, (0,0))[1] for fig in self._figs]
        cell_w = max(widths) if widths else 30
        cell_h = max(heights) if heights else 30

        # --- Single figure ---
        if n == 1:
            fig = self._figs[0]
            fw, fh = self._fig_sizes.get(fig, (cell_w, cell_h))
            self._canvas.config(width=fw, height=fh)
            target_w = min(fw, self._screen_w)
            target_h = min(fh, self._screen_h)
            self._root.geometry(f"{target_w}x{target_h}+0+{self._screen_h - target_h}")
            self._canvas.delete("all")
            fca = self._fcas.get(fig)
            if fca is not None:
                fca.draw()
                photo = tk.PhotoImage(master=self._canvas, width=int(fw), height=int(fh))
                tkagg.blit(photo, fca.get_renderer()._renderer, colormode=2)
                img_id = self._canvas.create_image(fw/2, fh/2, image=photo)
                self._photos[fig] = photo
                self._fig_ids[fig] = img_id
            self._root.update()
            return

        # --- 2 to 6 figures: improved layout ---
        if 2 <= n <= 6:
            # Better layout logic for side-by-side viewing
            if n == 2:
                cols = 2
                rows = 1
            elif n == 3:
                cols = 3
                rows = 1
            elif n == 4:
                cols = 2
                rows = 2
            elif n == 5 or n == 6:
                cols = 3
                rows = 2
            
            canvas_w = cols * cell_w
            canvas_h = rows * cell_h
            self._canvas.config(width=canvas_w, height=canvas_h)
            self._canvas.delete("all")
            self._fig_ids.clear()

            for idx, fig in enumerate(self._figs):
                col = idx % cols
                row = idx // cols
                x = col * cell_w + cell_w/2
                y = row * cell_h + cell_h/2
                fca = self._fcas.get(fig)
                if fca is None:
                    continue
                fca.draw()
                fw, fh = self._fig_sizes.get(fig, (cell_w, cell_h))
                photo = tk.PhotoImage(master=self._canvas, width=int(fw), height=int(fh))
                tkagg.blit(photo, fca.get_renderer()._renderer, colormode=2)
                img_id = self._canvas.create_image(x, y, image=photo)
                self._photos[fig] = photo
                self._fig_ids[fig] = img_id

            # Better window sizing to fit content
            target_w = min(canvas_w + 50, self._screen_w)
            target_h = min(canvas_h + 50, self._screen_h)
            self._root.geometry(f"{target_w}x{target_h}+0+{self._screen_h - target_h}")
            self._root.update()
            return

        # --- More than 6 figures: grid layout ---
        cols = 3  # 3 columns for many figures
        rows = math.ceil(n / cols)
        canvas_w = cols * cell_w
        canvas_h = rows * cell_h
        self._canvas.config(width=canvas_w, height=canvas_h)
        self._canvas.delete("all")
        self._fig_ids.clear()

        for idx, fig in enumerate(self._figs):
            col = idx % cols
            row = idx // cols
            x = col * cell_w + cell_w/2
            y = row * cell_h + cell_h/2
            fca = self._fcas.get(fig)
            if fca is None:
                continue
            fca.draw()
            fw, fh = self._fig_sizes.get(fig, (cell_w, cell_h))
            photo = tk.PhotoImage(master=self._canvas, width=int(fw), height=int(fh))
            tkagg.blit(photo, fca.get_renderer()._renderer, colormode=2)
            img_id = self._canvas.create_image(x, y, image=photo)
            self._photos[fig] = photo
            self._fig_ids[fig] = img_id

        target_w = min(canvas_w + 50, self._screen_w)
        target_h = min(canvas_h + 50, self._screen_h)
        self._root.geometry(f"{target_w}x{target_h}+0+{self._screen_h - target_h}")
        self._root.update()

    # --- Remaining LivePlotter methods ---
    def plot_figure(self, fig):
        if self._empty:
            self._empty = False
            try:
                self._canvas.delete(self._text_id)
            except Exception:
                pass
        fca = FigureCanvasAgg(fig)
        fca.draw()
        f_w, f_h = fca.get_renderer().get_canvas_width_height()
        f_w, f_h = int(f_w), int(f_h)
        self._figs.append(fig)
        self._fcas[fig] = fca
        self._fig_sizes[fig] = (f_w, f_h)
        self._rebuild_canvas_layout()

    def plot_new_figure(self):
        fig = plt.Figure(figsize=(3, 2), edgecolor="black")
        ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
        fig.tight_layout()
        self.plot_figure(fig)
        return fig, ax

    def plot_new_dynamic_figure(self, title=""):
        dyfig = DynamicFigure(figsize=(6,4.5), title=title)
        self.plot_figure(dyfig.fig)
        return dyfig

    def plot_new_dynamic_2d_figure(self, title="", **kwargs):
        if title and "trajectory" in title.lower():
            big_figsize = kwargs.pop("figsize", (15, 10))
            dy2dfig = Dynamic2DFigure(figsize=big_figsize, title=title, **kwargs)
        else:
            # Increased default size for better visibility
            default_figsize = kwargs.pop("figsize", (15, 10))
            dy2dfig = Dynamic2DFigure(figsize=default_figsize, title=title, **kwargs)
        self.plot_figure(dy2dfig.fig)
        return dy2dfig

    def refresh_figure(self, fig):
        if fig not in self._fcas:
            return
        fca = self._fcas[fig]
        fca.draw()
        photo = self._photos.get(fig)
        if photo is None:
            self._rebuild_canvas_layout()
            return
        tkagg.blit(photo, fca.get_renderer()._renderer, colormode=2)
        self._root.update()

    def init_pygame(self):
        self._game_frame = tk.Frame(self._root, width=self._surf_w, height=self._surf_h)
        self._game_frame.grid(row=0, column=1)
        os.environ['SDL_WINDOWID'] = str(self._game_frame.winfo_id())
        self._game_frame.update()
        pygame.display.init()

    def plot_surface(self, surf):
        s_w, s_h = surf.get_size()
        self._surf_w += s_w
        self._surf_h = max(self._surf_h, s_h)
        if not self._pygame_init:
            self._pygame_init = True
            self.init_pygame()
        else:
            self._game_frame.config(width=self._surf_w, height=self._surf_h)
            self._game_frame.grid(row=0, column=1)
        self._display = pygame.display.set_mode((self._surf_w, self._surf_h))
        self._surfs.append(surf)
        self._surf_coords[surf] = (self._surf_w-s_w, 0)
        self._display.blits(list(self._surf_coords.items()))

    def refresh(self):
        for fig in list(self._figs):
            if fig in self._fcas:
                self.refresh_figure(fig)
        self._root.update()
        if self._display is not None:
            self._display.blits(list(self._surf_coords.items()))
            pygame.display.flip()
