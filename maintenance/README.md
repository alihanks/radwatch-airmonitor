# maintenance/

Scripts for host-level (server OS) maintenance. Distinct from `image_scripts/`, which is the pipeline itself.

When the dosenet server has issues that aren't about CNF parsing, weather scraping, or plot generation — freezes, kernel messages, microcode warnings, package updates — start with `docs/runbook.md` §7 (Host system maintenance) for the diagnostic path, and use the scripts here to apply fixes.

## Scripts

| Script | Purpose |
|--------|---------|
| `update_microcode.sh` | Install/upgrade CPU microcode and report SRSO mitigation status. See `docs/runbook.md` §7 for the freeze diagnosis it addresses. |

Run scripts as `sudo bash maintenance/<script>.sh` from the repo root unless the script's own usage note says otherwise. Most include a `--check` mode that reports state without making changes.
