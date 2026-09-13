import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_zoo.core import models
for row in models():
    print(row['model_id'], row['group'], row['architecture'], row['lineage_id'], row['status'])

