#!/usr/bin/env python3
"""Escape the accented first-name initials of the .bib exported from Zotero, in LaTeX.

WHY THIS EXISTS. cas-model2-names.bst abbreviates the first name to its
initial, and in doing so TRUNCATES the multibyte UTF-8 character, keeping
only its first byte. The result is a main.bbl that does not decode as UTF-8:
with `pdflatex -halt-on-error` it produces no PDF, and without that option
it prints eight surnames with a broken initial. Checked with a minimal case:
the literal `Zeljko` gives `Ivezic, <byte>.`, and `{\\v Z}eljko` gives
`{\\v Z}.`, which is correct. Classic BibTeX is not Unicode-aware and treats
`{\\v Z}` as an opaque group, which is exactly what is needed.

WHY HERE AND NOT IN ZOTERO. Better BibTeX does export already escaped, but
it generates a DIFFERENT citation key (ivezicLSSTScienceDrivers2019 versus
ivezic_lsst_2019), and the manuscript cites 38 keys in the old format:
switching exporters would break all of them. This script leaves the
exporter and the keys as they are and fixes only what is broken.

IT ONLY TOUCHES FIRST NAMES inside `author` and `editor` fields, and only
when the initial is an accented letter. It does not touch surnames (they
are not abbreviated, so they are not truncated), nor titles, abstracts,
or URLs.

Usage: python fix_bib_initials.py ../GPUPHOT_manuscript/references_zotero.bib
       python fix_bib_initials.py --check <file>   # do not write, report only
"""
import re
import sys

# Accented letter -> its LaTeX escape. Only the ones that appear as the INITIAL of a first name.
ESCAPES = {
    'Ž': r'{\v Z}', 'É': r'{\'E}', 'Á': r'{\'A}', 'Ł': r'{\L}', 'İ': r'{\.I}',
    'Ó': r"{\'O}", 'Ö': r'{\"O}', 'Ü': r'{\"U}', 'Å': r'{\AA}', 'Ø': r'{\O}',
    'Č': r'{\v C}', 'Š': r'{\v S}', 'Ç': r'{\c C}', 'Ñ': r'{\~N}', 'Í': r"{\'I}",
    'Ú': r"{\'U}", 'À': r'{\`A}', 'È': r'{\`E}', 'Ê': r'{\^E}', 'Ä': r'{\"A}',
}
FIELDS = ('author', 'editor')


def fix_field(value):
    """Escape the accented first-name initial of each name in an author/editor field."""
    changes = []
    parts = value.split(' and ')
    for i, p in enumerate(parts):
        if ',' not in p:
            continue                      # "Collaboration X" with no comma: no first name
        surname, _, firstname = p.partition(',')
        n = firstname.lstrip()
        if not n:
            continue
        if n[0] in ESCAPES:
            new_name = ESCAPES[n[0]] + n[1:]
            parts[i] = surname + ', ' + new_name
            changes.append(f'{surname.strip()}, {n} -> {new_name}')
    return ' and '.join(parts), changes


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    check_only = '--check' in sys.argv
    if not args:
        raise SystemExit(__doc__)
    path = args[0]
    text = open(path, encoding='utf-8').read()

    all_changes = []

    def repl(m):
        value, changes = fix_field(m.group(2))
        all_changes.extend(changes)
        return m.group(1) + value + m.group(3)

    pattern = re.compile(r'(\b(?:' + '|'.join(FIELDS) + r')\s*=\s*\{)(.*?)(\}\s*,?\s*\n)',
                         re.S | re.I)
    new_text = pattern.sub(repl, text)

    if not all_changes:
        print('Nothing to fix: no first-name initials have accents.')
        return
    print(f'{len(all_changes)} initials to escape:')
    for c in sorted(set(all_changes)):
        print('  ', c)
    if check_only:
        print('\n--check: nothing written.')
        return
    open(path, 'w', encoding='utf-8').write(new_text)
    print(f'\nWritten {path}.')
    print('Reminder: run this again AFTER EVERY export from the reference manager, '
          'because Zotero overwrites the file.')


if __name__ == '__main__':
    main()
