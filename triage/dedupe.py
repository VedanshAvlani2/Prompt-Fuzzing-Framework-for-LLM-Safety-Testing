def dedupe(records):
    seen, out = set(), []
    for r in records:
        sig = (r["seed"], r["mutated"], r["response"])
        if sig not in seen:
            seen.add(sig); out.append(r)
    return out
