from config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    MODEL_DIR,
    NUMERIC_FEATURES,
    TARGET,
)


def test_config_values_are_defined():
    assert TARGET == "target"
    assert DATA_PATH.name == "dataset.csv"
    assert MODEL_DIR.name == "models"
    assert len(NUMERIC_FEATURES) > 0
    assert len(CATEGORICAL_FEATURES) > 0


def test_features_do_not_contain_target():
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert TARGET not in all_features
