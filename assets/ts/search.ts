interface SearchPage { title: string; permalink: string; content: string; }
const form = document.querySelector<HTMLFormElement>('.search-form[data-json]');
if (form) {
    const input = form.querySelector<HTMLInputElement>('input[name="keyword"]')!;
    const list = document.querySelector<HTMLElement>('.search-result--list')!;
    const status = document.querySelector<HTMLElement>('#search-status')!;
    let dataPromise: Promise<SearchPage[]> | undefined;
    let revision = 0;
    let timer: ReturnType<typeof setTimeout>;

    function loadData() {
        if (!dataPromise) {
            dataPromise = fetch(form!.dataset.json!).then(response => {
                if (!response.ok) throw new Error('Search index unavailable');
                return response.json();
            }).catch(error => { dataPromise = undefined; throw error; });
        }
        return dataPromise;
    }

    function highlight(element: HTMLElement, text: string, keywords: string[]) {
        const pattern = keywords.map(word => word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
        const regex = new RegExp(pattern, 'giu');
        let end = 0;
        for (const match of text.matchAll(regex)) {
            const start = match.index!;
            element.append(document.createTextNode(text.slice(end, start)));
            const mark = document.createElement('mark');
            mark.textContent = match[0];
            element.append(mark);
            end = start + match[0].length;
        }
        element.append(document.createTextNode(text.slice(end)));
    }

    async function search() {
        const current = ++revision;
        const query = input.value.trim();
        const url = new URL(location.href);
        query ? url.searchParams.set('keyword', query) : url.searchParams.delete('keyword');
        history.replaceState(null, '', url);
        list.replaceChildren();
        if (!query) { status.textContent = '输入关键词，搜索文章标题和正文。'; return; }
        status.textContent = '正在搜索…';
        const keywords = [...new Set(query.toLocaleLowerCase().split(/\s+/u))];
        try {
            const pages = await loadData();
            if (current !== revision) return;
            const results = pages.map(page => {
                const title = page.title.toLocaleLowerCase();
                const content = page.content.toLocaleLowerCase();
                const score = keywords.reduce((sum, word) => sum + (title.includes(word) ? 10 : 0) + (content.includes(word) ? 1 : 0), 0);
                return { page, score, matches: keywords.every(word => title.includes(word) || content.includes(word)) };
            }).filter(result => result.matches).sort((a, b) => b.score - a.score);
            status.textContent = results.length ? `找到 ${results.length} 篇相关文章` : `没有找到“${query}”，试试更短的关键词。`;
            const fragment = document.createDocumentFragment();
            for (const { page } of results) {
                const article = document.createElement('article');
                const link = document.createElement('a');
                link.href = page.permalink;
                const details = document.createElement('div');
                details.className = 'article-details';
                const title = document.createElement('h2');
                title.className = 'article-title';
                highlight(title, page.title, keywords);
                const excerpt = document.createElement('p');
                excerpt.className = 'article-preview';
                const positions = keywords.map(word => page.content.toLocaleLowerCase().indexOf(word)).filter(position => position >= 0);
                const start = Math.max(0, (positions.length ? Math.min(...positions) : 0) - 35);
                const text = (start ? '…' : '') + page.content.slice(start, start + 180) + (start + 180 < page.content.length ? '…' : '');
                highlight(excerpt, text, keywords);
                details.append(title, excerpt);
                link.append(details); article.append(link); fragment.append(article);
            }
            list.append(fragment);
        } catch {
            if (current === revision) status.textContent = '搜索索引暂时无法加载，请重新搜索，或到归档页浏览文章。';
        }
    }
    input.addEventListener('input', (event: InputEvent) => {
        clearTimeout(timer);
        ++revision; // Cancel stale results even while typing or using a Chinese IME.
        if (!event.isComposing) timer = setTimeout(search, 160);
    });
    input.addEventListener('compositionend', () => { clearTimeout(timer); timer = setTimeout(search, 160); });
    form.addEventListener('submit', event => { event.preventDefault(); clearTimeout(timer); search(); });
    window.addEventListener('popstate', () => { input.value = new URL(location.href).searchParams.get('keyword') || ''; search(); });
    input.value = new URL(location.href).searchParams.get('keyword') || '';
    search();
}
