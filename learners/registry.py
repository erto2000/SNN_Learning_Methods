# utils/registry.py
from learners.backprop import BackpropLearner
from learners.forward_forward import FFLearner
from learners.eprop import EpropLearner
from learners.pepita import PepitaLearner

LEARNER_REGISTRY = {
    "bp": BackpropLearner,
    "ff": FFLearner,
    "eprop": EpropLearner,
    "pepita": PepitaLearner,
}
