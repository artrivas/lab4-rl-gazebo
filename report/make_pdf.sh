#!/bin/bash
# HTML -> PDF con Microsoft Edge headless (Windows). Uso: report/make_pdf.sh [salida.pdf]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=${1:-$HERE/Informe_Lab4.pdf}
python3 "$HERE/build_report.py" "$HERE/informe_lab4.html"
mkdir -p /mnt/c/WSL/report && cp "$HERE/informe_lab4.html" /mnt/c/WSL/report/informe_lab4.html
EDGE="/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
cd /mnt/c/WSL/report
"$EDGE" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf='C:\WSL\report\Informe_Lab4.pdf' 'file:///C:/WSL/report/informe_lab4.html' 2>/dev/null || true
for i in $(seq 1 30); do [ -s /mnt/c/WSL/report/Informe_Lab4.pdf ] && break; sleep 1; done
cp /mnt/c/WSL/report/Informe_Lab4.pdf "$OUT"
echo "PDF: $OUT ($(du -h "$OUT" | cut -f1))"
