#!/usr/bin/env python3
"""Test the year-folder filter in sample_collection.build_collection_incremental.

This filter has been a repeat source of bugs because of confusion about the
on-disk layout. Pinning the expected behavior here with concrete paths so a
future change to the regex either keeps these passing or makes the
disagreement explicit.

The actual layout is:
    .../current/<YYYY>/<file>.CNF
where <YYYY> is a 4-digit year (e.g. 2025, 2026) and <file>.CNF typically
includes a date stamp inside the filename. Anything else under current/
(temp_test_folder/, scratch dirs, etc.) is rejected.

Run from the repo root:
    python3 test/test_date_dir_filter.py
"""
import re
import sys

# Keep this regex literal in sync with image_scripts/sample_collection.py.
DATE_DIR_RE = re.compile(r'/\d{4}/[^/]+$')

CASES = [
    # (path, should_match, description)
    ('/dropbox/.../current/2025/Roof_2025-08-15_001.CNF', True,
        'CNF in year folder, date in filename'),
    ('/dropbox/.../current/2026/Roof_2026-05-15_001.CNF', True,
        'recent year folder'),
    ('/dropbox/.../current/2026/no_date_in_name.CNF', True,
        'CNF in year folder, no date in filename (still accepted)'),
    ('/dropbox/.../current/temp_test_folder/Roof_2025-08-15_001.CNF', False,
        'CNF in non-year folder is rejected even with date in filename'),
    ('/dropbox/.../current/temp_test_folder/whatever.CNF', False,
        'CNF in non-year folder, no date'),
    ('/dropbox/.../current/2026/sub/Roof.CNF', False,
        'Three levels deep is rejected (glob does not reach it anyway)'),
    ('/dropbox/.../current/25/Roof.CNF', False,
        'Two-digit year folder rejected'),
    ('/dropbox/.../current/2026-05-15/Roof.CNF', False,
        'Date-named folder (not actually used here) rejected'),
    ('/dropbox/.../current/2026/', False,
        'Bare directory path (no file part) rejected'),
]

failed = 0
for path, want, desc in CASES:
    got = bool(DATE_DIR_RE.search(path))
    mark = 'OK' if got == want else 'FAIL'
    if got != want:
        failed += 1
    print(f'{mark:4s} want={str(want):5s} got={str(got):5s} {desc}')
    print(f'         {path}')

if failed:
    print(f'\n{failed} test(s) FAILED')
    sys.exit(1)
print('\nAll tests passed.')
