# Report build script

Generates `13.docx` (and, via LibreOffice, `13.pdf`) from the diagrams in `docs/diagrams/`
and the screenshots in `docs/screenshots/`. Re-run this after updating either.

```bash
cd docs/report_build
npm install docx
node build_report.js          # writes ../../13.docx

# optional: also produce 13.pdf
cd ../..
soffice --headless --convert-to pdf 13.docx
```

Before running, fill in the real BITS IDs, names and contribution percentages in the
`COVER` table at the top of `build_report.js` (currently `<BITS_ID_n>` / `<NAME_n>` placeholders).
