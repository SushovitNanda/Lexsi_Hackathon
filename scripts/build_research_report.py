"""Build the plain-language report as real LaTeX and compile it with Tectonic.

Requires pandoc, tectonic and pdftoppm (brew install pandoc tectonic poppler).
The Markdown is the content source; the generated .tex can also be compiled
directly. Build products and render PNGs stay in tmp/pdfs/ except .tex and .pdf.
"""
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
REPO = 'https://github.com/SushovitNanda/Lexsi_Hackathon'
BUNDLE = 'https://data1b.fullyjustified.net/tlextras-2022.0r0.tar'


def main():
    source = (ROOT / 'SIMPLE_RESEARCH_REPORT.md').read_text()
    source = source.split('\n', 1)[1].lstrip()
    # Use repository URLs so evidence links work in the shared PDF.
    def link(match):
        label, target = match.groups()
        if target.startswith(('https://', 'http://', '#')):
            return match.group(0)
        kind = 'tree' if (ROOT / target).is_dir() else 'blob'
        return f'[{label}]({REPO}/{kind}/manishshaw/{quote(target, safe="/")})'
    source = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, source)
    ast = json.loads(subprocess.run(['pandoc', '--from=markdown-smart', '--to=json'],
        input=source, text=True, check=True, capture_output=True).stdout)
    tables = 0
    laid_out = []
    for block in ast['blocks']:
        if block['t'] == 'Header':
            block['c'][0] -= 1
            if block['c'][2] and block['c'][2][0].get('c') == '6.':
                laid_out.append({'t': 'RawBlock', 'c': ['latex', '\\clearpage']})
        if block['t'] == 'Table':
            widths = [[.40, .12, .14, .21, .13], [.22, .34, .44], [.20, .38, .42]][tables]
            specs = block['c'][2]
            if len(specs) != len(widths):
                raise ValueError('Table layout changed; review column widths')
            for spec, width in zip(specs, widths):
                spec[1] = {'t': 'ColWidth', 'c': width}
            tables += 1
        laid_out.append(block)
    ast['blocks'] = laid_out
    if tables != 3:
        raise ValueError('Expected result, research-question and limitations tables')
    tex = subprocess.run(['pandoc', '--from=json', '--to=latex', '--standalone',
        '--template', str(ROOT / 'reports/research_report.template.tex')],
        input=json.dumps(ast), text=True, check=True, capture_output=True).stdout
    out = ROOT / 'output/pdf'
    temp = ROOT / 'tmp/pdfs'
    out.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    tex_path = out / 'manish_research_report.tex'
    tex_path.write_text(tex)
    subprocess.run(['tectonic', '--bundle', BUNDLE, '--keep-logs', '--outdir', str(temp), str(tex_path)], check=True)
    pdf = temp / 'manish_research_report.pdf'
    target = out / pdf.name
    target.write_bytes(pdf.read_bytes())
    subprocess.run(['pdftoppm', '-r', '120', '-png', str(target), str(temp / 'report-page')], check=True)
    print(f'PDF: {target}\nLaTeX: {tex_path}\nRender pages: {temp}')


if __name__ == '__main__':
    main()
