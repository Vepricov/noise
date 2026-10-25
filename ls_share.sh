#!/bin/bash
# Is our part of the shares still there: the step-1000 checkpoint and the 30M data, on nfs2 and nfs3.
for d in /home/jovyan/shares/SR006.nfs2 /home/jovyan/shares/SR006.nfs3; do
  echo "=== $d"; df -h "$d" | tail -1; ls "$d" | head -30
done
ls -la /home/jovyan/shares/SR006.nfs2/veprikov/noise/ /home/jovyan/shares/SR006.nfs2/veprikov/noise/o30-base-a 2>&1 | head -20
ls -la /home/jovyan/shares/SR006.nfs3/veprikov 2>&1 | head
