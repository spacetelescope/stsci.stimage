# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'xy_coord_match'
author = f"Space Telescope Science Institute (STScI)"
copyright = f"{datetime.datetime.today().year}, Association of Universities for Research in Astronomy (AURA)"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx_automodapi.automodapi"]

templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "_static/stsci_pri_combo_mark_dark_bkgd.png"
html_last_updated_fmt = "%b %d, %Y"
html_sidebars = {"**": ["globaltoc.html", "relations.html", "searchbox.html"]}
html_domain_indices = True
html_use_index = True
