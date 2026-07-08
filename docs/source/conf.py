import os
import sys
sys.path.insert(0, os.path.abspath("../.."))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 't-roo'
copyright = '2026, Anna Puecher, Hauke Koehn'
author = 'Anna Puecher, Hauke Koehn'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]


autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "imported-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "exclude-members": "__weakref__",
}

autodoc_typehints = "description"
autodoc_preserve_defaults = True
autodoc_typehints = "description"
autodoc_typehints_format = "short"



source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_logo = "images/t-roo_name.png"

html_theme_options = {
    "navigation_with_keys": True,
    "sidebar_hide_name": False,
}

html_static_path = ['_static']
