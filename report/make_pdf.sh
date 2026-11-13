#!/bin/bash
# HTML -> PDF con Microsoft Edge headless (Windows). Uso: report/make_pdf.sh [salida.pdf]
# Usa una carpeta temporal en C:\WSL que se borra al terminar.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=${1:-$HERE/Informe_Lab4.pdf}
TMP=/mnt/c/WSL/_report_tmp
python3 "$HERE/build_report.py" "$HERE/informe_lab4.html"
rm -rf "$TMP" && mkdir -p "$TMP" && cp "$HERE/informe_lab4.html" "$TMP/informe_lab4.html"
EDGE="/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
cd "$TMP"
"$EDGE" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf='C:\WSL\_report_tmp\Informe_Lab4.pdf' 'file:///C:/WSL/_report_tmp/informe_lab4.html' 2>/dev/null || true
for i in $(seq 1 30); do [ -s "$TMP/Informe_Lab4.pdf" ] && break; sleep 1; done
cp "$TMP/Informe_Lab4.pdf" "$OUT"
cd / && rm -rf "$TMP"
echo "PDF: $OUT ($(du -h "$OUT" | cut -f1))"
