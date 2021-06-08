import usingfym
import baseline
import time


t0 = time.time()
usingfym.run()
print(f"Using Fym: {time.time() - t0} seconds")

t0 = time.time()
baseline.run()
print(f"Baseline: {time.time() - t0} seconds")
