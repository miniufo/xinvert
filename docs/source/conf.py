# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys
import datetime
from docutils import nodes
sys.path.append(os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../'))
import xinvert


# -- Project information -----------------------------------------------------

project = 'xinvert'
copyright = f'{datetime.datetime.today().year}, MiniUFO'
author = 'MiniUFO'

# The full version, including alpha/beta/rc tags
version = xinvert.__version__
# The full version, including alpha/beta/rc tags
release = xinvert.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',     # api auto-gen
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',     # math
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'nbsphinx',
    'numpydoc',
    #'recommonmark',           # Markdown
    #'myst_nb',
    #'myst_parser',            # Markdown
]

# The master toctree document.
# master_doc = 'index'

# The suffix(es) of source filenames, either a string or list.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    'conf.py', 'sphinxext', '_build', '_templates', '_themes',
    '**.ipynb_checkpoints' '.DS_Store', 'trash', 'tmp',
]


# -- Options for nbsphinx ----------------------------------------------------
# Do not execute notebooks during the docs build: the examples need the
# bundled ``Data/`` directory (absent on ReadTheDocs) and some of them need a
# CUDA GPU.  The stored outputs are used verbatim.
nbsphinx_execute = 'never'


# -- Options for HTML output -------------------------------------------------

# Logo
html_logo = os.path.join('_static', 'xinvertLogo.png')

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
    'collapse_navigation': True,
    'navigation_depth': 4,
    'prev_next_buttons_location': 'bottom',  # top and bottom
}
html_css_files = [
    'my_theme.css',
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large. Static folder is for CSS and image files. Use ImageMagick to
# convert png to ico on command line with 'convert image.png image.ico'
html_favicon = os.path.join('_static', 'xinvertIcon.ico')


# -- Workarounds for two upstream rendering bugs ------------------------------
#
# Kept in sync with xdispersion/docs/source/conf.py and
# xcontour/docs/source/conf.py.  The only difference between the three copies is
# the package name inside the `subprocess.Popen' guard attribute, e.g.
# `_xinvert_wide_columns' here.  Copy this block verbatim into a new package.
#
# 1) MathJax 4 rejects AMS environments nested inside `split`.
#
#    `sphinx.ext.mathjax.html_visit_displaymath` wraps *every* display-math
#    block that contains a bare ``\\`` in ``\begin{split}...\end{split}``.  A
#    multi-row equation that already brings its own ``align`` environment ends
#    up as ``split > align``.  MathJax 3 tolerated that; MathJax 4 -- the
#    version Sphinx >= 8.2 loads from the CDN and the one ReadTheDocs serves --
#    aborts with "Erroneous nesting of equation structures" and renders
#    nothing.  A block that already carries an AMS environment needs no
#    wrapper, so it is emitted verbatim.  A multi-row block that carries
#    ``\tag`` *without* an environment of its own gets one supplied, because
#    ``split`` also rejects ``\tag not allowed in split environment``
#    (MathJax 3 and 4 alike).  Together these two branches mean notebook maths
#    can be written in whatever style the author prefers.
#
# 2) pandoc folds long lines, and docutils then cuts the `.. math::' short.
#
#    nbsphinx calls pandoc twice: markdown -> json (no width option at all) and
#    json -> rst with a hard-coded ``--columns=500`` ("Avoid breaks in tables,
#    see issue #240").  Anything longer than its column is soft-wrapped, and
#    pandoc keeps the indentation of a continuation line only when it breaks at
#    whitespace.  A break inside a macro produces an unindented continuation,
#    docutils therefore closes the ``.. math::`` directive early, and the tail
#    degrades into a definition list -- the equation shows up truncated.
#
#    A bare ``---`` line makes this much worse, and it appears in these
#    notebooks as a plain horizontal rule:
#
#      * between two ``---`` lines, pandoc sees a *single-column multiline
#        table* and swallows everything in between into one cell, then wraps
#        that cell at roughly 5.5% of ``--columns`` (500 -> 26 chars,
#        1000 -> 54, 5000 -> 279).  A 72-character equation is folded even at
#        --columns=1000, because 54 < 72.
#      * at the very start of a cell, pandoc reads ``---`` as a YAML metadata
#        block, exits 64 and returns an empty stdout, losing the whole cell.
#
#    Widening the columns helps but is a moving target, so pandoc is also told
#    ``--wrap=none``: with no folding at all there is nothing for docutils to
#    trip over, whatever the author writes.

_AMS_ENV = re.compile(
    r'\\begin\{(?:align|alignat|flalign|gather|multline|eqnarray|equation)\*?\}'
)

#: ``\tag`` is rejected by ``split``/``aligned``; it needs a real AMS environment.
_TAG = re.compile(r'\\tag\b')

#: pandoc's table-cell wrapping width (upstream nbsphinx uses 500).
PANDOC_COLUMNS = 5000

#: pandoc's line wrapping; 'none' disables it altogether.
PANDOC_WRAP = 'none'


def _visit_displaymath(self, node):
    """MathJax-4-safe ``html_visit_displaymath`` (see workaround 1 above)."""
    from sphinx.locale import _
    from sphinx.util.math import get_node_equation_number

    self.body.append(self.starttag(node, 'div', CLASS='math notranslate nohighlight'))
    equation = node.astext()
    if node.get('no-wrap', node.get('nowrap', False)):
        # the author supplied the delimiters together with `:nowrap:`
        self.body.append(self.encode(equation))
        self.body.append('</div>')
        raise nodes.SkipNode

    # necessary to e.g. set the id property correctly
    if node['number']:
        number = get_node_equation_number(self, node)
        self.body.append('<span class="eqno">(%s)' % number)
        self.add_permalink_ref(node, _('Link to this equation'))
        self.body.append('</span>')
    self.body.append(self.builder.config.mathjax_display[0])
    if _AMS_ENV.search(equation):
        # already an AMS environment -- the `split' wrapper below would nest it
        self.body.append(self.encode(equation))
    elif r'\\' in equation and _TAG.search(equation):
        # multi-line block carrying `\tag' but no environment of its own: the
        # `split' wrapper rejects `\tag not allowed in split environment', so
        # supply the align environment the numbering needs.
        parts = [prt for prt in equation.split('\n\n') if prt.strip()]
        joined = r' \\ '.join(parts)
        self.body.append(r'\begin{align}' + self.encode(joined) + r'\end{align}')
    else:
        parts = [prt for prt in equation.split('\n\n') if prt.strip()]
        if len(parts) > 1:  # Add alignment if there are more than 1 equation
            self.body.append(r' \begin{align}\begin{aligned}')
        for i, part in enumerate(parts):
            part = self.encode(part)
            if r'\\' in part:
                self.body.append(r'\begin{split}' + part + r'\end{split}')
            else:
                self.body.append(part)
            if i < len(parts) - 1:  # append new line if not the last equation
                self.body.append(r'\\')
        if len(parts) > 1:  # Add alignment if there are more than 1 equation
            self.body.append(r'\end{aligned}\end{align} ')
    self.body.append(self.builder.config.mathjax_display[1])
    self.body.append('</div>\n')
    raise nodes.SkipNode


def _widen_pandoc_columns(columns=PANDOC_COLUMNS, wrap=PANDOC_WRAP):
    """Stop pandoc from folding notebook content (see workaround 2 above).

    Only the json -> rst pandoc call is touched -- the one that carries
    ``--columns``.  Every other subprocess in the build passes through
    untouched.
    """
    import subprocess
    from os.path import basename

    if getattr(subprocess.Popen, '_xinvert_wide_columns', False):
        return

    real_popen = subprocess.Popen

    def _popen(cmd, *args, **kwargs):
        if (isinstance(cmd, (list, tuple)) and cmd
                and basename(str(cmd[0])).lower().startswith('pandoc')
                and any(str(arg).startswith('--columns=') for arg in cmd)):
            cmd = [f'--columns={columns}' if str(arg).startswith('--columns=')
                   else arg for arg in cmd]
            if wrap and not any(str(arg).startswith('--wrap') for arg in cmd):
                cmd.append(f'--wrap={wrap}')
        return real_popen(cmd, *args, **kwargs)

    _popen._xinvert_wide_columns = True
    subprocess.Popen = _popen


def setup(app):
    """Register the workarounds above (conf.py doubles as a local extension)."""
    # (1) replace the block-math renderer registered by sphinx.ext.mathjax
    #     (add_html_math_renderer refuses to overwrite, so poke the registry)
    app.registry.html_block_math_renderers['mathjax'] = (_visit_displaymath, None)

    # (2) stop pandoc from folding long lines
    _widen_pandoc_columns()

    return {'parallel_read_safe': True}
