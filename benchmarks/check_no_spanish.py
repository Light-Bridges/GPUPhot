#!/usr/bin/env python3
"""Flag Spanish text in tracked files before a push.

WHY: the repository is meant to ship with the paper, so its data headers, comments and
docstrings have to read in English. A find-and-replace pass is not enough: a header that
is deleted instead of translated also passes a Spanish check, and that is the failure this
script is written against. It reports what is STILL Spanish; it cannot see what was removed.

WHAT IT CHECKS: tracked files only, comment lines and prose cells. Accented characters
alone are not evidence (author names, object names), so it matches on function words that
do not occur in English technical prose.

USAGE:  python3 benchmarks/check_no_spanish.py [--all]
        exit code 1 if anything is found, so it can gate a push.
"""
import os
import re
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTS = ('.py', '.sh', '.csv', '.md', '.tex', '.yml', '.yaml', '.txt')
# Function words that do not appear in English technical prose. Deliberately narrow:
# a false positive costs a look, a false negative ships.
PAL = (r'\b(que|para|pero|porque|cuando|donde|desde|hasta|entre|sobre|segun|aunque|'
       r'este|esta|esto|esos|esas|cada|todos|todas|solo|mismo|misma|hay|son|fue|era|'
       r'tiene|hace|puede|debe|sale|queda|sin|con|por|del|las|los|una|uno)\b')
RE = re.compile(PAL, re.IGNORECASE)
# This file is skipped: it carries the Spanish function words as its own pattern,
# so it would always report itself.
SALTAR = ('benchmarks/data/raw/', 'dev/cleancode/',
          'benchmarks/check_no_spanish.py')


def tracked():
    out = subprocess.run(['git', 'ls-files'], cwd=BASE, capture_output=True, text=True)
    return [f for f in out.stdout.split('\n') if f.endswith(EXTS)]


def main():
    todo = '--all' in sys.argv
    hits = []
    for f in tracked():
        if not todo and any(f.startswith(s) for s in SALTAR):
            continue
        p = os.path.join(BASE, f)
        try:
            txt = open(p, encoding='utf-8', errors='replace').read()
        except OSError:
            continue
        for n, line in enumerate(txt.split('\n'), 1):
            if len(RE.findall(line)) >= 3:      # 3+ function words: prose, not a stray word
                hits.append((f, n, line.strip()[:100]))
    for f, n, l in hits[:60]:
        print(f'  {f}:{n}: {l}')
    if len(hits) > 60:
        print(f'  ... y {len(hits) - 60} mas')
    print(f'\n{len(hits)} lines look like Spanish prose, in '
          f'{len({h[0] for h in hits})} files.')
    print('REMINDER: translate, do not delete. A removed header also passes this check,\n'
          'and the caveat it carried is what stops the data being misused.')
    return 1 if hits else 0


if __name__ == '__main__':
    sys.exit(main())
