with open("src/patienttriage/audit.py", "r") as f:
    content = f.read()

new_method = """
    def clear_all(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM audit_events")
            try:
                conn.execute("DELETE FROM sqlite_sequence WHERE name='audit_events'")
            except sqlite3.OperationalError:
                pass # sqlite_sequence might not exist if table is empty
        self.last_hash = "GENESIS"
"""

if "def clear_all(self):" not in content:
    content += new_method
    with open("src/patienttriage/audit.py", "w") as f:
        f.write(content)
