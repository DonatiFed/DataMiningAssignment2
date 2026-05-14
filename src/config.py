from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"

TRAIN_FILE = DATA_DIR / "training_set_VU_DM.csv"
TEST_FILE = DATA_DIR / "test_set_VU_DM.csv"
SAMPLE_SUB_FILE = DATA_DIR / "submission_sample.csv"

TRAIN_ONLY_COLS = ["position", "click_bool", "booking_bool", "gross_bookings_usd"]

ID_COLS = ["srch_id", "prop_id"]

NON_FEATURE_COLS = [
    "srch_id", "prop_id", "date_time",
    "position", "click_bool", "booking_bool", "gross_bookings_usd",
]

RANDOM_SEED = 42
VAL_FRACTION = 0.1
