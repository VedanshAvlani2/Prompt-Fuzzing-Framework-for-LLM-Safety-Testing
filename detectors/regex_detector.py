# import re
# class RegexDetector:
#     def __init__(self):
#         self.rules = {"illegal": 8, "bypass": 7, "exploit": 9, "harm": 10}

#     def score(self, text):
#         matches, sev = [], 0
#         for k, s in self.rules.items():
#             if re.search(k, text, re.I):
#                 matches.append({"keyword": k, "severity": s})
#                 sev = max(sev, s)
#         return {"matches": matches, "severity": sev, "is_unsafe": sev >= 7}
