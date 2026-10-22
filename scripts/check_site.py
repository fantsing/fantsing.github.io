"""Validate a built Hugo site using only Python's standard library.

Usage: python scripts/check_site.py public
Checks local assets/links/fragments, search coverage, duplicate URLs, and metadata.
"""
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit, unquote
import json
import re
import sys
import xml.etree.ElementTree as ET


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.links, self.ids, self.images, self.headings = [], [], [], []
        self.canonical = self.description = self.lang = ''
        self.redirect = self.article = False
        self.text = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        if tag == 'html':
            self.lang = attrs.get('lang')
        if tag == 'meta':
            if attrs.get('http-equiv', '').lower() == 'refresh':
                self.redirect = True
            if attrs.get('name') == 'description':
                self.description = attrs.get('content')
        if tag == 'link' and attrs.get('rel') == 'canonical':
            self.canonical = attrs.get('href')
        if tag == 'article' and 'main-article' in attrs.get('class', '').split():
            self.article = True
        if tag == 'h1':
            self.headings.append(attrs)
        if tag == 'img':
            self.images.append(attrs)
        attr = {'a': 'href', 'img': 'src', 'script': 'src', 'link': 'href', 'source': 'src'}.get(tag)
        if attr and attrs.get(attr):
            self.links.append((tag, attrs[attr]))
        for source in attrs.get('srcset', '').split(','):
            if source.strip():
                self.links.append(('source', source.strip().split()[0]))

    def handle_data(self, text):
        self.text.append(text)


def check(root):
    root = root.resolve()
    docs = {file.relative_to(root).as_posix(): Document(file.read_text(encoding='utf-8'))
            for file in root.rglob('*.html')}
    files = {file.relative_to(root).as_posix() for file in root.rglob('*') if file.is_file()}
    errors, references, images = [], 0, 0
    canonical = docs['index.html'].canonical
    origin = urlsplit(canonical).netloc

    def resolve(url):
        path = unquote(urlsplit(url).path).lstrip('/')
        if not path or path.endswith('/'):
            path += 'index.html'
        elif path not in files and path + '/index.html' in files:
            path += '/index.html'
        return path

    for path, doc in docs.items():
        if doc.redirect:
            continue
        if not doc.lang or not doc.description or not doc.canonical:
            errors.append(f'{path}: missing language, description, or canonical metadata')
        if doc.article and len(doc.headings) != 1:
            errors.append(f'{path}: expected one article h1, got {len(doc.headings)}')
        duplicates = [name for name, count in Counter(doc.ids).items() if count > 1]
        if duplicates:
            errors.append(f'{path}: duplicate ids {duplicates}')
        text = ''.join(doc.text)
        if re.search(r'!\[[^\]]*\]\([^\n]*\.(?:png|jpg|webp)', text):
            errors.append(f'{path}: unrendered Markdown image')
        if '<h1 class="article-title">' in text or '<h2 class="article-title">' in text:
            errors.append(f'{path}: escaped heading markup')
        for image in doc.images:
            images += 1
            if 'alt' not in image:
                errors.append(f'{path}: image missing alt text')
        for tag, href in doc.links:
            url = urljoin(canonical, '/' + path)
            url = urljoin(url, href)
            parts = urlsplit(url)
            if parts.scheme not in ('http', 'https') or parts.netloc != origin:
                continue
            references += 1
            target = resolve(url)
            if target not in files:
                errors.append(f'{path}: broken {tag} {href}')
            elif parts.fragment and target in docs and unquote(parts.fragment) not in docs[target].ids:
                errors.append(f'{path}: missing anchor {href}')

    index = json.loads((root / 'search/index.json').read_text(encoding='utf-8'))
    urls = [entry['permalink'] for entry in index]
    if len(set(urls)) != len(urls):
        errors.append('Search index contains duplicate article URLs')
    article_paths = {path for path, doc in docs.items() if doc.article and path.startswith('p/')}
    indexed_paths = {resolve(url) for url in urls}
    if article_paths != indexed_paths:
        errors.append(f'Search coverage differs: {article_paths ^ indexed_paths}')
    # Catch source pages overwritten by colliding permalinks.
    sources = Path('content/post')
    if sources.is_dir():
        expected = 0
        for source in sources.rglob('index.md'):
            text = source.read_text(encoding='utf-8')
            front = text.split('+++', 2)[1] if text.startswith('+++') else text.split('---', 2)[1]
            if not re.search(r'(?:draft|hidden)\s*[:=]\s*true', front):
                expected += 1
        if len(index) != expected:
            errors.append(f'{expected} source posts but {len(index)} indexed articles')
    for file in root.rglob('*.css'):
        for href in re.findall(r'url\([\'"]?([^\)\'\"]+)', file.read_text(encoding='utf-8')):
            if href.startswith(('data:', 'http:', 'https:', '#')):
                continue
            url = urljoin(canonical, '/' + file.relative_to(root).as_posix())
            if resolve(urljoin(url, href)) not in files:
                errors.append(f'{file.relative_to(root)}: missing CSS asset {href}')
    for file in root.rglob('*.xml'):
        try:
            ET.parse(file)
        except ET.ParseError as error:
            errors.append(f'{file.relative_to(root)}: invalid XML: {error}')
    if not (root / 'robots.txt').is_file():
        errors.append('Missing robots.txt')
    report = {'html_pages': len(docs), 'articles': len(index), 'local_references': references,
              'image_references': images, 'errors': sorted(set(errors))}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(check(Path(sys.argv[1] if len(sys.argv) > 1 else 'public')))
