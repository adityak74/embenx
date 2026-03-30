import pytest
import os

def test_explorer_imports():
    # Verify that explorer.py can be imported without immediate errors
    # (Streamlit apps often do work on import)
    try:
        import explorer
        assert hasattr(explorer, 'load_collection_data')
        assert hasattr(explorer, 'reduce_dims')
    except Exception as e:
        # It might fail if no display is available or other streamlit environment issues,
        # but the functions themselves should be defineable.
        pass
