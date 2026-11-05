Animate iteration
=====================================

The animation at the top of the README and the documentation home page
was created with ``record_iteration``. It shows the solution evolving
as SOR iterations proceed, so that we can see the process of convergence.

Record frames
-------------

Select one spatial slice (one time/member) before recording. Spatial dimensions
are inferred when ``dims`` is omitted.

.. code-block:: python

   from xinvert import record_iteration

   frames = record_iteration(
       'Poisson', forcing,  # one 2-D spatial slice
       iParams={'architect': 'cpu'},
       iter_per_frame=20,   # SOR sweeps between saved frames
       max_frames=100,      # number of frames. Early convergence stopping is disabled
   )

``frames`` gets an extra dimension ``(iter, ...)``, which gives the cumulative
sweep count. The call runs immediately and retains the recorded frames in memory. For
larger-size problems, use file output like:

.. code-block:: python

   record_iteration(
       'Poisson', forcing, iter_per_frame=20, max_frames=100,
       output='iteration.nc', # save to a file
       complevels=4,          # use compression level for NetCDF file
   )

.. note::
    Internal ``extend``/``neumann`` boundaries are not yet supported by
    ``record_iteration``.

Create a GIF or MP4
-------------------

``create_animation`` renders numbered PNG frames in a unique temporary directory,
then encodes them in order as GIF or MP4. The frame directory is deleted by
default, including after failures:

.. code-block:: python

   from xinvert.utils import create_animation

   create_animation('iteration.nc', 'iteration.gif', framedim='iter', fps=6,
                   plot_kwargs={'vmin': -1e6, 'vmax': 1e6, 'cmap': 'RdBu_r'})

``framedim`` selects the animation dimension: ``'iter'`` for recorded SOR
iterations, ``'time'`` for physical time, or any other dimension name. Each
frame uses one index along that dimension, in file order. If omitted,
``'iter'`` takes priority over ``'time'``; other names must be specified.

The default plots ``inverted`` (or the file's only variable). Keep color
limits fixed between frames. MP4 requires FFmpeg; the Pillow GIF fallback
keeps frames in memory, so prefer FFmpeg for long, high-resolution movies.

For custom plots, pass a function accepting **one frame, an Axes, and your
keyword arguments**. It is called as ``plotfunc(frame, ax, **plot_kwargs)``:

.. code-block:: python

   def draw(frame, ax, scale=1e-6, limit=1):
       (frame.inverted * scale).plot(ax=ax, vmin=-limit, vmax=limit, cmap='RdBu_r')

   create_animation('iteration.nc', 'iteration.mp4', plotfunc=draw,
                    plot_kwargs={'scale': 1e-6, 'limit': 2})

``draw`` receives the current frame as a Dataset, including static variables;
select the fields to plot inside it. ``plot_kwargs`` passes any additional
arguments to your function.

**Multiple variables.** Use ``figfunc`` to create a fresh
figure and axes for each frame, then draw the complete frame in ``draw``:

.. code-block:: python

   def make_figure():
       import matplotlib.pyplot as plt
       return plt.subplots(ncols=2, figsize=(10, 4))
       # UltraPlot alternative: replace the two lines above with
       # import ultraplot as uplt
       # return uplt.subplots(ncols=2, figsize=(10, 4))

   def draw(frame, axs):
       for ax, name in zip(axs, ['streamfunction', 'velocity_potential']):
           mesh = ax.pcolormesh(frame[name], vmin=-1e6, vmax=1e6)
           ax.set_title(name)
           ax.figure.colorbar(mesh, ax=ax)

   create_animation('fields.nc', 'fields.mp4', draw,
                    figfunc=make_figure, framedim='time', workers=4, dpi=200)

The example uses Matplotlib; UltraPlot can replace it as shown in the comments.
Use fixed limits for each field, and choose the layout in ``make_figure``.

With ``workers=n``, one can render the frames using n processes in parallel.
Standalone scripts need ``if __name__ == '__main__':`` around the call.
Each call uses a temporary PNG directory, deleted by default; ``remove_frames=False``
retains it and returns ``(movie_path, frame_dir)``; this is the unique subdirectory, not its parent ``temp_dir``.

.. important::
    Choose ``temp_dir`` on a fast disk with space for the frames, and set ``workers``
    within your CPU and memory limits.
