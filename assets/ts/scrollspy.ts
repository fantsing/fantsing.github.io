// Track only real TOC links; Hugo can emit empty list items for skipped heading levels.
export function setupScrollspy() {
    const navigation = document.querySelector<HTMLElement>('#TableOfContents');
    const article = document.querySelector<HTMLElement>('.article-content');
    if (!navigation || !article) return;
    const sections = [...navigation.querySelectorAll<HTMLAnchorElement>('li > a[href^="#"]')].flatMap(link => {
        let id: string;
        try { id = decodeURIComponent(link.hash.slice(1)); } catch { return []; }
        const heading = document.getElementById(id);
        return heading ? [{ heading, item: link.parentElement!, link, offset: 0 }] : [];
    });
    if (!sections.length) return;
    let active: HTMLElement | undefined;
    let frame = 0;
    function update() {
        frame = 0;
        let current: typeof sections[number] | undefined;
        for (const section of sections) {
            if (section.offset <= window.scrollY + 60) current = section;
        }
        if (current?.item === active) return;
        active?.classList.remove('active-class');
        active = current?.item;
        active?.classList.add('active-class');
        if (current && navigation!.offsetParent && !navigation!.matches(':hover, :focus-within')) {
            const offset = current.link.getBoundingClientRect().top - navigation!.getBoundingClientRect().top;
            if (offset < 0 || offset > navigation!.clientHeight - 40) {
                navigation!.scrollTo({ top: navigation!.scrollTop + offset - navigation!.clientHeight / 2, behavior: 'auto' });
            }
        }
    }
    function measure() {
        for (const section of sections) section.offset = section.heading.getBoundingClientRect().top + window.scrollY;
        update();
    }
    window.addEventListener('scroll', () => { if (!frame) frame = requestAnimationFrame(update); }, { passive: true });
    window.addEventListener('resize', measure);
    new ResizeObserver(measure).observe(article);
    measure();
}
