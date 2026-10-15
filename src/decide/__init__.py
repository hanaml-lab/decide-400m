"""decide-400m: typed answers about a state in one encoder pass, calibrated, with a bound on the error rate of the
decisions it makes and a deferral for the rest."""
from .calibrate import Calibration, ece, fit_temperatures
from .model import Answer, DecisionModel, ask
from .packing import Choice, Question, Score, YesNo, pack
from .risk import decision_threshold, state_threshold

__all__ = ["Answer", "Calibration", "Choice", "DecisionModel", "Question", "Score", "YesNo", "ask",
           "decision_threshold", "ece", "fit_temperatures", "pack", "state_threshold"]
__version__ = "0.1.0"
