import sqlite3
import json
import time

DB = r"C:\Users\童珏玮\.local\share\mimocode\mimocode.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

# 1. Recent sessions (last 30 days)
print("=== RECENT SESSIONS (last 30 days) ===")
cutoff = int((time.time() - 30 * 86400) * 1000)
cur = conn.execute(
    "SELECT id, time_created, title, directory, project_id FROM session WHERE time_created > ? ORDER BY time_created DESC",
    (cutoff,),
)
sessions = []
for r in cur.fetchall():
    sessions.append(dict(r))
    print(f"  {r['id']} | t={r['time_created']} | {r['title']} | {r['directory']}")

print(f"\nTotal sessions: {len(sessions)}")

# 2. Get project IDs to filter
print("\n=== PROJECTS ===")
cur = conn.execute("SELECT id, path, name FROM project")
for r in cur.fetchall():
    print(f"  {r['id']} | {r['path']} | {r['name']}")

# 3. Repeated tool usage across all sessions
print("\n=== REPEATED TOOL USAGE (all sessions) ===")
cur = conn.execute("""
    SELECT json_extract(p.data, '$.tool') as tool,
           substr(json_extract(p.data, '$.state.input'), 1, 150) as input_preview,
           count(*) as n,
           count(DISTINCT m.session_id) as session_count
    FROM message m
    JOIN part p ON p.message_id = m.id
    WHERE json_extract(m.data, '$.role') = 'assistant'
      AND json_extract(p.data, '$.type') = 'tool'
    GROUP BY tool, input_preview
    HAVING n >= 2
    ORDER BY n DESC
    LIMIT 40
""")
for r in cur.fetchall():
    print(f"  [{r['n']}x in {r['session_count']} sessions] {r['tool']}: {r['input_preview'][:100]}")

# 4. User turns with repeated keywords
print("\n=== USER TURNS WITH REPEATED KEYWORDS ===")
cur = conn.execute("""
    SELECT m.session_id, substr(json_extract(m.data, '$.content'), 1, 200) as text
    FROM message m
    WHERE json_extract(m.data, '$.role') = 'user'
      AND (
        lower(json_extract(m.data, '$.content')) LIKE '%again%'
        OR lower(json_extract(m.data, '$.content')) LIKE '%every time%'
        OR lower(json_extract(m.data, '$.content')) LIKE '%repeat%'
        OR lower(json_extract(m.data, '$.content')) LIKE '%same as before%'
        OR lower(json_extract(m.data, '$.content')) LIKE '%the usual%'
        OR lower(json_extract(m.data, '$.content')) LIKE '%like last%'
        OR lower(json_extract(m.data, '$.content')) LIKE '%always%'
        OR lower(json_extract(m.data, '$.content')) LIKE '%每次%'
        OR lower(json_extract(m.data, '$.content')) LIKE '%重复%'
        OR lower(json_extract(m.data, '$.content')) LIKE '%跟上次%'
        OR lower(json_extract(m.data, '$.content')) LIKE '%跟以前%'
      )
    LIMIT 30
""")
for r in cur.fetchall():
    print(f"  [{r['session_id']}] {r['text'][:150]}")

# 5. Repeated file paths in tool calls
print("\n=== REPEATED FILE PATHS IN TOOL CALLS ===")
cur = conn.execute("""
    SELECT json_extract(p.data, '$.tool') as tool,
           substr(json_extract(p.data, '$.state.input'), 1, 200) as input_preview,
           count(*) as n
    FROM message m
    JOIN part p ON p.message_id = m.id
    WHERE json_extract(m.data, '$.role') = 'assistant'
      AND json_extract(p.data, '$.type') = 'tool'
      AND json_extract(p.data, '$.tool') IN ('read_file', 'write_file', 'search_files', 'list_files')
    GROUP BY tool, input_preview
    HAVING n >= 3
    ORDER BY n DESC
    LIMIT 30
""")
for r in cur.fetchall():
    print(f"  [{r['n']}x] {r['tool']}: {r['input_preview'][:150]}")

# 6. Repeated command patterns (bash/terminal)
print("\n=== REPEATED BASH COMMANDS ===")
cur = conn.execute("""
    SELECT substr(json_extract(p.data, '$.state.input'), 1, 200) as cmd,
           count(*) as n,
           count(DISTINCT m.session_id) as session_count
    FROM message m
    JOIN part p ON p.message_id = m.id
    WHERE json_extract(m.data, '$.role') = 'assistant'
      AND json_extract(p.data, '$.type') = 'tool'
      AND json_extract(p.data, '$.tool') = 'bash'
    GROUP BY cmd
    HAVING n >= 2
    ORDER BY n DESC
    LIMIT 30
""")
for r in cur.fetchall():
    print(f"  [{r['n']}x in {r['session_count']} sessions] {r['cmd'][:150]}")

conn.close()
