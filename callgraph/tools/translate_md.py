#!/usr/bin/env python3
"""Translate Markdown files under `callgraph` into English and write them to `callgraph_en`.

Behavior:
- Preserva bloques de código (```...```) y fragmentos inline en backticks by default.
- Optionally, translate JSON-like fenced code blocks (--translate-code) by parsing JSON and translating string values.
- Intenta usar `deep-translator` (GoogleTranslator) if installed.
- Si no está disponible, intenta usar la utilidad de sistema `trans` (translate-shell) si está instalada.
- Si ninguna herramienta de traducción automática está disponible, copia los ficheros y añade un marcador "[TRANSLATION_NEEDED]" al principio del archivo destino para indicar que necesita traducción manual o instalar dependencias.

Usage:
    python tools/translate_md.py --src callgraph --dst callgraph_en --translate-code

Nota: Revisar manualmente las traducciones para terminología técnica.
"""
from pathlib import Path
import re
import argparse
import subprocess
import sys
import json
from typing import Any

# Try dynamic import of deep_translator later if needed
_HAS_DEEP = False
GoogleTranslator = None


def try_import_deep():
    """Attempt to import deep_translator.GoogleTranslator and set globals.
    This is performed lazily so the package can be installed between runs.
    """
    global _HAS_DEEP, GoogleTranslator
    if _HAS_DEEP:
        return
    try:
        from deep_translator import GoogleTranslator as _GT
        GoogleTranslator = _GT
        _HAS_DEEP = True
    except Exception:
        _HAS_DEEP = False

CODE_FENCE_RE = re.compile(r'(```[\s\S]*?```)', re.MULTILINE)
INLINE_CODE_RE = re.compile(r'`([^`]+)`')
JSON_FENCE_RE = re.compile(r'^```(?:\w+)?\n([\s\S]*?)\n```$', re.MULTILINE)

CHUNK_SIZE = 4000


def translate_with_deep(text: str) -> str:
    # ensure translator imported
    try_import_deep()
    if not _HAS_DEEP or GoogleTranslator is None:
        raise RuntimeError('deep-translator not available')
    return GoogleTranslator(source='auto', target='en').translate(text)


def translate_with_trans_shell(text: str) -> str:
    # use translate-shell: `trans -b :en`
    p = subprocess.Popen(['trans', '-b', ':en'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate(text)
    if p.returncode != 0:
        raise RuntimeError(f"trans failed: {err.strip()}")
    return out


def translate_text(text: str) -> str:
    if not text.strip():
        return text
    # attempt deep translator dynamically
    try:
        try_import_deep()
        if _HAS_DEEP:
            try:
                return translate_with_deep(text)
            except Exception as e:
                print(f"deep-translator failed: {e}", file=sys.stderr)
    except Exception:
        pass
    # try translate-shell
    try:
        return translate_with_trans_shell(text)
    except Exception:
        # fallback: no translator available
        return None


def translate_json_strings(obj: Any) -> Any:
    # Walk the JSON-like structure and translate string leaf nodes.
    if isinstance(obj, dict):
        return {k: translate_json_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [translate_json_strings(v) for v in obj]
    elif isinstance(obj, str):
        # translate the string content
        res = translate_text(obj)
        return res if res is not None else obj
    else:
        return obj


def translate_code_fence_if_json(fence_text: str, translate_code: bool) -> str:
    # fence_text includes opening and closing ``` lines. If translate_code is True and inner looks like JSON, try to parse and translate string values.
    if not translate_code:
        return fence_text
    m = JSON_FENCE_RE.match(fence_text)
    if not m:
        return fence_text
    inner = m.group(1)
    stripped = inner.strip()
    if not stripped:
        return fence_text
    if not (stripped.startswith('{') or stripped.startswith('[')):
        return fence_text
    try:
        parsed = json.loads(inner)
    except Exception:
        return fence_text
    translated_parsed = translate_json_strings(parsed)
    pretty = json.dumps(translated_parsed, ensure_ascii=False, indent=2)
    # rebuild fence preserving possible language tag from opening line
    opening = fence_text.split('\n', 1)[0]
    return f"{opening}\n{pretty}\n```"


def preserve_inline_code_and_translate(segment: str) -> str:
    # segment: plain text (no code fences)
    inline_codes = []

    def repl(m):
        idx = len(inline_codes)
        inline_codes.append(m.group(0))
        return f"__INLINE_CODE_{idx}__"

    placeholdered = INLINE_CODE_RE.sub(repl, segment)

    # Translate placeholdered text in chunks
    translated_parts = []
    i = 0
    while i < len(placeholdered):
        chunk = placeholdered[i:i+CHUNK_SIZE]
        # try avoid splitting mid-sentence by expanding to next newline if possible
        if i + CHUNK_SIZE < len(placeholdered):
            nxt = placeholdered.find('\n', i + CHUNK_SIZE)
            if nxt != -1 and nxt - i <= CHUNK_SIZE + 200:
                chunk = placeholdered[i:nxt]
                i = nxt
            else:
                i += CHUNK_SIZE
        else:
            i += CHUNK_SIZE
        res = translate_text(chunk)
        if res is None:
            return None
        translated_parts.append(res)

    translated = ''.join(translated_parts)

    # restore inline codes
    def restore(m):
        idx = int(m.group(1))
        return inline_codes[idx]

    translated = re.sub(r'__INLINE_CODE_(\d+)__', restore, translated)
    return translated


def translate_markdown(md_text: str, translate_code: bool) -> (str, bool):
    # Returns (translated_text, success_flag). If success_flag is False then translation failed.
    segments = CODE_FENCE_RE.split(md_text)
    out_segments = []
    for seg in segments:
        if seg.startswith('```'):
            # optionally translate JSON-like fences
            newf = translate_code_fence_if_json(seg, translate_code)
            out_segments.append(newf)
        else:
            translated = preserve_inline_code_and_translate(seg)
            if translated is None:
                return (None, False)
            out_segments.append(translated)
    return (''.join(out_segments), True)


def process_file(src_path: Path, dst_path: Path, translate_code: bool) -> bool:
    text = src_path.read_text(encoding='utf-8')
    translated, ok = translate_markdown(text, translate_code)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if ok:
        dst_path.write_text(translated, encoding='utf-8')
    else:
        # fallback: copy and add marker
        marker = "[TRANSLATION_NEEDED]\n\n"
        dst_path.write_text(marker + text, encoding='utf-8')
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='callgraph')
    parser.add_argument('--dst', default='callgraph_en')
    parser.add_argument('--translate-code', action='store_true', help='Also translate JSON-like fenced code blocks')
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        print(f"Source {src_root} does not exist", file=sys.stderr)
        sys.exit(2)

    md_files = list(src_root.rglob('*.md'))
    if not md_files:
        print("No markdown files found in source directory.")
        sys.exit(0)

    print(f"Found {len(md_files)} .md files. Translating to {dst_root}...")
    all_ok = True
    for f in md_files:
        rel = f.relative_to(src_root)
        dst = dst_root / rel
        ok = process_file(f, dst, args.translate_code)
        print(f"Processed: {f} -> {dst} {'(translated)' if ok else '(marker added)'}")
        all_ok = all_ok and ok

    # final note about deep-translator availability
    try_import_deep()
    if not _HAS_DEEP:
        print('\nNote: deep-translator not installed. Install with: python3 -m pip install --user deep-translator')
    print('\nDone.')
    if not all_ok:
        print('Some files could not be auto-translated; they were copied with a marker. Install translation tooling for full automatic translations.')


if __name__ == '__main__':
    main()
