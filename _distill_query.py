import sqlite3
import json
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DB = r"C:\Users\童珏玮\.local\share\mimocode\mimocode.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cutoff = int((time.time() - 30 * 86400) * 1000)

# Debug: check message data structure
print("=== MESSAGE DATA STRUCTURE (sample) ===")
cur = conn.execute("""
    SELECT data FROM message WHERE data IS NOT NULL LIMIT 3
""")
for r in cur.fetchall():
    d = json.loads(r['data'])
    print(json.dumps(d, indent=2, ensure_ascii=False)[:500])
    print("---")

# Try different content field names
print("\n=== USER MESSAGE CONTENT VARIANTS ===")
cur = conn.execute("""
    SELECT m.session_id, m.data, s.title
    FROM message m
    JOIN session s ON s.id = m.session_id
    WHERE json_extract(m.data, '$.role') = 'user'
      AND s.title NOT LIKE '%checkpoint%'
    LIMIT 5
""")
for r in cur.fetchall():
    d = json.loads(r['data'])
    print(f"  keys: {list(d.keys())}")
    # Try various content fields
    for key in ['content', 'text', 'message', 'input', 'prompt']:
        if key in d:
            val = d[key]
            if isinstance(val, str) and len(val) > 5:
                print(f"  [{key}] {val[:200]}")
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and 'text' in item:
                        print(f"  [{key}].text {item['text'][:200]}")
    print("---")

# Also check parts for user messages
print("\n=== PARTS FOR USER MESSAGES ===")
cur = conn.execute("""
    SELECT p.data, m.session_id, s.title
    FROM part p
    JOIN message m ON p.message_id = m.id
    JOIN session s ON s.id = m.session_id
    WHERE json_extract(m.data, '$.role') = 'user'
      AND s.title NOT LIKE '%checkpoint%'
    LIMIT 5
""")
for r in cur.fetchall():
    d = json.loads(r['data'])
    print(f"  part keys: {list(d.keys())}")
    if 'text' in d:
        print(f"  text: {d['text'][:200]}")
    print("---")

conn.close()
