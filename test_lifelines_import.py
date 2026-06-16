import lifelines
print(lifelines.__version__)
try:
    from lifelines.utils import brier_score
    print("found in utils")
except ImportError:
    print("not found in utils")

try:
    from lifelines.scoring import brier_score
    print("found in scoring")
except ImportError:
    print("not found in scoring")

try:
    from lifelines.brier_score import brier_score
    print("found in brier_score")
except ImportError:
    print("not found in brier_score")

try:
    import lifelines.brier_score
    print("found brier_score module")
except ImportError:
    print("not found brier_score module")
