from .base import DriftStream, Generator
from .classification import CLASSIFICATION_GENERATORS, SEA, LED, Agrawal, Hyperplane, Mixed, RandomRBF, RandomTree, STAGGER, Waveform
from .regression import REGRESSION_GENERATORS, Friedman, HyperplaneRegression, Mv, Planes2D
from .simulated import SIMULATED_GENERATORS, PoissonDemand, RegimeSwitchingWalk, SeasonalSensor

GENERATORS = {**CLASSIFICATION_GENERATORS, **REGRESSION_GENERATORS, **SIMULATED_GENERATORS}

__all__ = ["Generator", "DriftStream", "GENERATORS"] + list(GENERATORS)
