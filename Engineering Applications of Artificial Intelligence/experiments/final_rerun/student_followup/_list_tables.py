import re
from pathlib import Path

base = Path(r"C:\Users\ruben\OneDrive\Desktop\BGU\THESIS\Engineering Applications of Artificial Intelligence\paper_supervisor_draft")
files = ["sec_intro", "sec_related", "sec_framework", "sec_instantiation", "sec_design",
         "sec_results", "sec_discussion", "sec_conclusion"]
i = 0
for fn in files:
    p = base / (fn + ".tex")
    if not p.exists():
        continue
    src = p.read_text(encoding="utf-8")
    for m in re.finditer(r"\\label\{tab:([^}]+)\}", src):
        i += 1
        print(f"Table {i}: tab:{m.group(1)}  ({fn}.tex)")
