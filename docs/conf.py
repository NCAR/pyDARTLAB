# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

src_path = os.path.abspath('../src')
sys.path.insert(0, src_path)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pydartlab'
copyright = '2024, University Corporation for Atmospheric Research'
author = 'Helen Kershaw'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_options = {
    'github_user': 'NCAR',
    'github_repo': 'pyDARTLAB',
    'github_button': 'true',
    'github_type': 'star',
    'fixed_sidebar': 'true',
    'sidebar_collapse': 'true',
    'sidebar_width': '325px',
    'page_width': '1200px',
    'show_powered_by' : 'false',
    'description': 'A Python version of DART_LAB, the interactive ensemble data assimilation tutorial from the Data Assimilation Research Testbed (DART).',
    'caption_font_size': '1.5em',
}


extensions = [
    'sphinx_copybutton',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
]

# Widget/plotting dependencies are not needed to build the docs
autodoc_mock_imports = ['ipywidgets', 'ipympl']

