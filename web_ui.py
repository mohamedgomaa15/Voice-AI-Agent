from flask import Flask, request, render_template_string, redirect, url_for, jsonify
from extract_entities import HybridIntentSystem

app = Flask(__name__)
system = HybridIntentSystem()
history = []

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hybrid Intent Extractor</title>
  <style>
    :root{--bg:#f4f7fb;--card:#ffffff;--muted:#6b7280;--accent:#2563eb}
    body{font-family:Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; background:var(--bg); color:#111; margin:0; padding:2rem}
    .wrap{max-width:900px;margin:0 auto}
    header{display:flex;align-items:center;gap:1rem;margin-bottom:1rem}
    h1{font-size:1.25rem;margin:0}
    form{display:flex;gap:.5rem;background:var(--card);padding:1rem;border-radius:12px;box-shadow:0 6px 18px rgba(16,24,40,0.06)}
    input[type=text]{flex:1;padding:.6rem .8rem;border:1px solid #e6edf3;border-radius:8px;font-size:1rem}
    input[type=submit]{background:var(--accent);color:#fff;border:none;padding:.6rem 1rem;border-radius:8px;cursor:pointer}
    a.clear{align-self:center;color:var(--muted);text-decoration:none;font-size:.9rem}
    .card{background:var(--card);padding:1rem;border-radius:10px;box-shadow:0 6px 18px rgba(16,24,40,0.04);margin-top:1rem}
    pre{white-space:pre-wrap;margin:0;font-family:inherit}
    .history{display:flex;flex-direction:column;gap:.6rem}
    .item{display:flex;justify-content:space-between;gap:1rem;padding:.75rem;border-radius:8px;background:#fbfdff;border:1px solid #eef5ff}
    .meta{color:var(--muted);font-size:.9rem}
    .small{font-size:.85rem;color:var(--muted)}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Hybrid Intent Extractor</h1>
      <div class="small">Local demo</div>
    </header>

    <form method="post" action="/">
      <input type="text" name="query" placeholder="Enter query (English or Arabic)" autofocus required>
      <input type="submit" value="Process">
      <a class="clear" href="/clear">Clear history</a>
    </form>

    {% if last %}
      <div class="card">
        <h3 style="margin-top:0">Last result</h3>
        <div><strong>Query:</strong> {{ last['query'] }}</div>
        <div class="meta">Intent: <strong>{{ last['intent'] }}</strong> — Confidence: {{ '%.2f'|format(last['confidence']*100) }}%</div>
        <div style="margin-top:.5rem"><strong>Entities:</strong>
          <pre>{{ last['entities'] }}</pre>
        </div>
        <div class="small" style="margin-top:.5rem">Processing time: {{ '%.2f'|format(last['timing_ms']) }} ms</div>
      </div>
    {% endif %}

    {% if history %}
      <div class="card" style="margin-top:1rem">
        <h3 style="margin-top:0">History</h3>
        <div class="history">
        {% for item in history %}
          <div class="item">
            <div>
              <div><strong>{{ item['query'] }}</strong></div>
              <div class="meta">Intent: {{ item['intent'] }} — {{ '%.2f'|format(item['confidence']*100) }}% • Entities: {{ item['entities'] }}</div>
            </div>
            <div class="small">{{ '%.0f'|format(item['timing_ms']) }} ms</div>
          </div>
        {% endfor %}
        </div>
      </div>
    {% endif %}
  </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    last = None
    if request.method == 'POST':
        q = request.form.get('query', '').strip()
        if q:
            res = system.process_query(q)
            entry = {
                'query': q,
                'intent': res['intent'],
                'confidence': res['confidence'],
                'entities': res['entities'],
                'timing_ms': res['timing']['total_ms']
            }
            history.insert(0, entry)
            last = entry
    return render_template_string(TEMPLATE, last=last, history=history)

@app.route('/api/query', methods=['POST'])
def api_query():
    data = request.get_json(force=True)
    q = data.get('query') if isinstance(data, dict) else None
    if not q:
        return jsonify({'error': 'no query provided'}), 400
    res = system.process_query(q)
    return jsonify(res)

@app.route('/clear')
def clear_history():
    history.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Flask web UI for HybridIntentSystem')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=5000, type=int)
    args = parser.parse_args()
    print(f"Starting web UI on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port)
