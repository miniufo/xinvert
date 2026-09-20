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
# 1) MathJax 4 rejects AMS environments nested inside `split`.
#
#    `sphinx.ext.mathjax.html_visit_displaymath` wraps *every* display-math
#    block that contains a bare ``\\`` in ``\begin{split}...\end{split}``.  A
#    multi-row equation in these notebooks already brings its own ``align``
#    environment, so the extra wrapper produces ``split > align``.  MathJax 3
#    tolerated that; MathJax 4 -- the version Sphinx >= 8.2 loads from the CDN
#    and the one ReadTheDocs serves -- aborts with
#    "Erroneous nesting of equation structures" and renders nothing.
#    A block that already carries an AMS environment needs no wrapper, so it is
#    emitted verbatim.  Affects every multi-row equation in the notebooks.
#
# 2) `nbsphinx.pandoc` appends a hard-coded ``--columns=500`` to its
#    json -> rst pass (nbsphinx/__init__.py, "Avoid breaks in tables, see
#    issue #240").  That width soft-wraps long equations inside markdown
#    tables, and pandoc indents the continuation line *only when it breaks at
#    whitespace*.  When the break lands inside a macro the continuation starts
#    at the cell's left edge, docutils therefore closes the ``.. math::``
#    directive early, and the equation is truncated -- the
#    "PV inversion for balanced flow" row of 00_Introduction.ipynb kept only
#    ``\frac{\pa`` of its 202-character equation.  Widening the wrapping so
#    that cells fit on a single line cures it without touching content.

_AMS_ENV = re.compile(
    r'\\begin\{(?:align|alignat|flalign|gather|multline|eqnarray|equation)\*?\}'
)

#: pandoc's table-cell wrapping width (upstream nbsphinx uses 500).
PANDOC_COLUMNS = 1000


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


def _widen_pandoc_columns(columns=PANDOC_COLUMNS):
    """Give pandoc more room when it renders markdown tables to reST.

    Only pandoc invocations match the predicate below; every other subprocess in
    the build is passed through untouched.
    """
    import subprocess
    from os.path import basename

    if not columns or getattr(subprocess.Popen, '_xinvert_wide_columns', False):
        return

    real_popen = subprocess.Popen

    def _popen(cmd, *args, **kwargs):
        if (isinstance(cmd, (list, tuple)) and cmd
                and basename(str(cmd[0])).lower().startswith('pandoc')):
            cmd = [f'--columns={columns}' if str(arg).startswith('--columns=') else arg
                   for arg in cmd]
        return real_popen(cmd, *args, **kwargs)

    _popen._xinvert_wide_columns = True
    subprocess.Popen = _popen


def setup(app):
    """Register the workarounds above (conf.py doubles as a local extension)."""
    # (1) replace the block-math renderer registered by sphinx.ext.mathjax
    #     (add_html_math_renderer refuses to overwrite, so poke the registry)
    app.registry.html_block_math_renderers['mathjax'] = (_visit_displaymath, None)

    # (2) widen pandoc's table-cell wrapping
    _widen_pandoc_columns()

    return {'parallel_read_safe': True}
