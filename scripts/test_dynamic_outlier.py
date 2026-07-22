from ice_offline.config.paths import table_path

import test_dynamic


DEFAULT_OUTPUT = table_path("dynamic", "success_outlier.csv")
OUTLIER_RATIO = 0.05


if __name__ == "__main__":
    test_dynamic.main(default_output=DEFAULT_OUTPUT, default_outlier_ratio=OUTLIER_RATIO)
