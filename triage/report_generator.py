from pathlib import Path
import json

def generate_html(json_path, out_path=None):
    data = json.load(open(json_path))
    rows = "".join(
        f"<tr><td>{d['seed']}</td><td>{d['mutated']}</td>"
        f"<td>{d['response']}</td><td>{d['is_unsafe']}</td><td>{d['severity']}</td></tr>"
        for d in data
    )
    html = f"""
    <html><body><h2>Prompt Fuzzer Report</h2>
    <table border=1><tr><th>Seed</th><th>Mutated</th><th>Response</th><th>Unsafe</th><th>Severity</th></tr>{rows}</table>
    </body></html>
    """
    out_path = out_path or Path(json_path).with_suffix(".html")
    Path(out_path).write_text(html)
    print(f"📄 Report saved to {out_path}")
