
import pytest

from data import list_zoo


def test_zoo_listing():
    datasets = list_zoo()
    assert "squad-v2" in datasets
    assert "ms-marco" in datasets
    assert len(datasets) >= 3

def test_zoo_load_invalid():
    from data import load_from_zoo
    with pytest.raises(ValueError, match="not found in zoo"):
        load_from_zoo("ghost-dataset")
