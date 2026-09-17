#!/usr/bin/env python3
"""Copies of the figures renumbered IN ORDER OF APPEARANCE, for submission.

THE PROBLEM. The generators name each figure after its content and the order
they were written in (fig1..fig8), but the manuscript includes them in a
different order and does not include all of them. Today `fig4_heatmap` is
Figure 3, `fig3_memory_comparison` is Figure 4, `fig7` is 5, `fig8` is 6,
`fig5` is 7, and `fig6_concurrency` is not included. In an editorial
production pipeline a mismatch like this is costly: someone matches file to
number by name and places the wrong figure.

THE DECISION. Nothing gets renamed. Just as the key in the GPU label map
still reads 'Orin NX 8GB (nvgpu)' because that is what the CSVs carry, the
canonical files keep their names because that is what the `.tex`, the
scripts and the reproduction paths cite. What this script does is emit
renumbered COPIES in a separate directory, plus the equivalence map, to
attach to the submission.

THE ORDER IS READ FROM THE .tex, not hardcoded here: if a section gets
reordered tomorrow, this stays correct without anyone remembering to
update it.

Output: GPUPHOT_manuscript/figures_submission/ (Figure_N.<ext>) and figure_name_map.csv
"""
import os
import re
import shutil
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(BASE, 'GPUPHOT_manuscript')
FIGS = os.path.join(MAN, 'figures')
OUT = os.path.join(MAN, 'figures_submission')
# Section order exactly as main.tex includes them.
ORDEN_TEX = ['introduction', 'architecture', 'implementation', 'deployment',
             'validation', 'performance', 'discussion', 'conclusion']
RE_INC = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}')


def apariciones():
    """(nombre_de_fichero, seccion) en el orden en que el documento las incluye."""
    out = []
    for sec in ORDEN_TEX:
        path = os.path.join(MAN, f'{sec}.tex')
        if not os.path.exists(path):
            continue
        for m in RE_INC.finditer(open(path, encoding='utf-8', errors='replace').read()):
            out.append((os.path.basename(m.group(1)), sec))
    return out


def main():
    ap = apariciones()
    if not ap:
        raise SystemExit('no se encontro ningun \\includegraphics; ¿cambio el nombre de las secciones?')
    os.makedirs(OUT, exist_ok=True)
    for viejo in os.listdir(OUT):
        os.remove(os.path.join(OUT, viejo))
    filas = ['figura_en_el_articulo,fichero_canonico,fichero_para_envio,seccion,coincide_el_numero']
    for n, (nombre, sec) in enumerate(ap, start=1):
        src = None
        for ext in ('.pdf', '.png'):
            cand = os.path.join(FIGS, nombre + ext)
            if os.path.exists(cand):
                src = cand
                break
        if src is None:
            print(f'  AVISO: Figura {n} ({nombre}) no tiene fichero en figures/; se omite')
            continue
        ext = os.path.splitext(src)[1]
        dst = os.path.join(OUT, f'Figure_{n}{ext}')
        shutil.copy2(src, dst)
        m = re.match(r'fig(\d+)_', nombre)
        coincide = 'si' if (m and int(m.group(1)) == n) else 'NO'
        filas.append(f'{n},{nombre}{ext},Figure_{n}{ext},{sec},{coincide}')
        print(f'  Figura {n:>2}  <-  {nombre}{ext:<5}  ({sec})  {"" if coincide=="si" else "<- el numero del nombre NO coincide"}')
    # the ones the generator produces that the paper does not use
    usados = {n for n, _ in ap}
    sobran = sorted({os.path.splitext(f)[0] for f in os.listdir(FIGS)
                     if f.endswith(('.pdf', '.png')) and not f.startswith('.')} - usados)
    sobran = [s for s in sobran if re.match(r'fig\d+_', s) and '.bak' not in s and 'bak_' not in s]
    if sobran:
        print(f'\n  generadas y NO incluidas en el articulo: {", ".join(sobran)}')
        for s in sobran:
            filas.append(f',{s},,,no_incluida')
    mapa = os.path.join(OUT, 'figure_name_map.csv')
    with open(mapa, 'w') as fh:
        fh.write('# Mapping between manuscript figure numbers and canonical generator output filenames.\n'
                 '# Canonical filenames are retained in benchmarks/ and referenced across LaTeX sources.\n'
                 '# Reproduce: python3 benchmarks/build_submission_figures.py\n')
        fh.write('\n'.join(filas) + '\n')
    print(f'\nescrito {OUT}/  ({len(ap)} figuras) y figure_name_map.csv')


if __name__ == '__main__':
    main()
