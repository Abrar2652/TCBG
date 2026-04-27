"""
TCBG: Temporal Curvature Bifiltration Graphcodes
Core source package.
"""
from .curvature import compute_forman_ricci, auto_epsilon
from .bifiltration import build_bifiltration
from .graphcode import compute_graphcode
from .pipeline import TCBGPipeline

def _lazy_gin():
    from .gin_classifier import GINClassifier, build_gin
    return GINClassifier, build_gin

__all__ = [
    'compute_forman_ricci',
    'auto_epsilon',
    'build_bifiltration',
    'compute_graphcode',
    'GINClassifier',
    'build_gin',
    'TCBGPipeline',
]
