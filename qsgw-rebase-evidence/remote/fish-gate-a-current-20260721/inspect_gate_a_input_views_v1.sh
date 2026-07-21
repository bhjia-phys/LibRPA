#!/usr/bin/env bash
set -u

BUNDLE=/home/bhj/ai-runs/librpa-qsgw-gate-a-symmetry-bundle-20260720-v3/dataset
COMPOSITE=/home/bhj/ai-runs/librpa-qsgw-gate2-current-20260722-aca53374-v1/input-overlay

printf '%s\n' '=== sha256 ==='
sha256sum \
  "$BUNDLE/stru_out" \
  "$COMPOSITE/stru_out" \
  "$BUNDLE/qsgw_input.contract" \
  "$COMPOSITE/qsgw_input.contract"

printf '%s\n' '=== line counts ==='
wc -l "$BUNDLE/stru_out" "$COMPOSITE/stru_out"

printf '%s\n' '=== source tail ==='
tail -n 35 "$BUNDLE/stru_out"

printf '%s\n' '=== composite tail ==='
tail -n 70 "$COMPOSITE/stru_out"

printf '%s\n' '=== diff ==='
diff -u "$BUNDLE/stru_out" "$COMPOSITE/stru_out" || true

printf '%s\n' '=== key files ==='
for p in \
  sks1k1_nao.txt sks1k2_nao.txt sks1k3_nao.txt sks1k4_nao.txt \
  sks1k5_nao.txt sks1k6_nao.txt sks1k7_nao.txt sks1k8_nao.txt \
  vxcs1k1_nao.txt vxcs1k2_nao.txt vxcs1k3_nao.txt vxcs1k4_nao.txt \
  vxcs1k5_nao.txt vxcs1k6_nao.txt vxcs1k7_nao.txt vxcs1k8_nao.txt \
  vxc_out
do
  if test -e "$BUNDLE/$p"; then
    stat -c '%n %s' "$BUNDLE/$p"
  else
    printf 'MISSING %s\n' "$BUNDLE/$p"
  fi
done

printf '%s\n' '=== hf/xc/s/vxc patterns ==='
find "$BUNDLE" -maxdepth 1 -type f -printf '%f\n' \
  | grep -E '^(hf_exchange|xc_matr|vxcs|sks)' \
  | sort
