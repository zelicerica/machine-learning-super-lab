"""Lesson-specific, annotated primary reading shared by the site and workbook."""
import json
from pathlib import Path

REFERENCES = json.loads(Path(__file__).with_name('lesson_references.json').read_text(encoding='utf-8'))

def references_markdown(lesson_id):
    sources = REFERENCES[lesson_id]
    if len(sources) < 2:
        raise ValueError(f'{lesson_id} needs multiple relevant references')
    lines = ['## References for this lesson', '',
             'Use these sources to check the mechanism and extend the exercise. Research results are not results measured by this lab.', '']
    lines.extend(f"- [{s['title']}]({s['url']}) — {s['why']}" for s in sources)
    return '\n'.join(lines) + '\n'
