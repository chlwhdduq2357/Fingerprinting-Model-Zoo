import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import csv
from itertools import combinations
from model_zoo.core import ROOT, models
from model_zoo import pair_relation

rows = [r for r in models() if r['status']=='verified']
path = ROOT/'metadata/pairs.csv'
with path.open('w',newline='',encoding='utf-8') as f:
    w = csv.DictWriter(f,fieldnames=['model_id_1','model_id_2','same_lineage','same_architecture','same_family','different_architecture','hard_negative'])
    w.writeheader()
    for a,b in combinations(rows,2):
        w.writerow(dict(model_id_1=a['model_id'],model_id_2=b['model_id'],**pair_relation(a,b)))
print(path)

