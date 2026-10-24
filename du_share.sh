#!/bin/bash
# What fills the shared disks: df and the largest folders of SR006.nfs2 and nfs3, two levels down.
for d in /home/jovyan/shares/SR006.nfs2 /home/jovyan/shares/SR006.nfs3; do
  echo "=== df $d"; df -h "$d" | tail -1
  echo "=== du $d"; du -xhd1 "$d" 2>/dev/null | sort -rh | head -25
done
echo "=== du nfs2 level 2"; du -xhd2 /home/jovyan/shares/SR006.nfs2 2>/dev/null | sort -rh | head -40
echo "=== veprikov"; du -sh /home/jovyan/shares/SR006.nfs2/veprikov/noise/* 2>/dev/null | sort -rh | head -30
