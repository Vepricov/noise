#!/bin/bash
# Frees our part of SR006.nfs2: every run folder under veprikov/noise except the step-1000 base checkpoint
# (o30-base-a) and the prepared data (data30m). Prints what it removes and df before and after.
set -u
ROOT=/home/jovyan/shares/SR006.nfs2/veprikov/noise
df -h /home/jovyan/shares/SR006.nfs2 | tail -1
du -sh "$ROOT"/* 2>/dev/null | sort -rh
for d in "$ROOT"/*; do
  case "$(basename "$d")" in o30-base-a|data30m) echo "keep $d";; *) echo "rm $d"; rm -rf "$d";; esac
done
df -h /home/jovyan/shares/SR006.nfs2 | tail -1
