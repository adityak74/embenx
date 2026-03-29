import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'embenx'
copyright = '2026, Aditya Karnam'
author = 'Aditya Karnam'
version = '0.0.1'
release = '0.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'furo'
html_static_path = ['_static']
html_title = "Embenx Documentation"
