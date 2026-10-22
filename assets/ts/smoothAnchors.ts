export function setupSmoothAnchors() {
    document.querySelectorAll<HTMLAnchorElement>('a[href^="#"]:not(.skip-link)').forEach(link => {
        link.addEventListener('click', event => {
            if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
            let id: string;
            try { id = decodeURIComponent(link.hash.slice(1)); } catch { return; }
            const target = document.getElementById(id);
            if (!target) return;
            event.preventDefault();
            history.pushState(null, '', link.hash);
            const top = target.getBoundingClientRect().top + window.scrollY - 24;
            window.scrollTo({ top, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
        });
    });
}
