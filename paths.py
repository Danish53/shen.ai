from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PATH = BACKEND_DIR / "linknet.pth"
DEEPPHYS_WEIGHTS_PATH = BACKEND_DIR / "weights" / "PURE_DeepPhys.pth"
DEEPPHYS_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/ubicomplab/rPPG-Toolbox/main/"
    "final_model_release/PURE_DeepPhys.pth"
)
OUTPUT_DIR = BACKEND_DIR / "output"
IMAGES_DIR = BACKEND_DIR / "images"

PULSE_PATH = OUTPUT_DIR / "pulse.npy"
HRS_PATH = OUTPUT_DIR / "hrs.npy"
BRS_PATH = OUTPUT_DIR / "brs.npy"
FFT_SPEC_PATH = OUTPUT_DIR / "fft_spec.npy"
RESULTS_PNG_PATH = OUTPUT_DIR / "results.png"

OUTPUT_DIR.mkdir(exist_ok=True)
