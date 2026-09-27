def test_class_names_count():
    from api.constants import CLASS_NAMES
    assert len(CLASS_NAMES) == 5

def test_class_names_are_unique():
    from api.constants import CLASS_NAMES
    assert len(CLASS_NAMES) == len(set(CLASS_NAMES))
