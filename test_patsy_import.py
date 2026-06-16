import traceback
try:
    from patsy import Q
    print("Successfully imported Q from patsy")
except ImportError as e:
    print(f"Failed to import Q from patsy: {e}")
    traceback.print_exc()