// Keyboard support and state announcements for the theme's existing controls.
const darkToggle = document.getElementById('dark-mode-toggle');
if (darkToggle) {
    darkToggle.setAttribute('role', 'button');
    darkToggle.tabIndex = 0;
    darkToggle.setAttribute('aria-label', '切换深色模式');
    const updateScheme = () => darkToggle.setAttribute('aria-pressed', String(document.documentElement.dataset.scheme === 'dark'));
    updateScheme();
    window.addEventListener('onColorSchemeChange', updateScheme);
    darkToggle.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); darkToggle.click(); }
    });
}
const menuToggle = document.getElementById('toggle-menu');
if (menuToggle) {
    menuToggle.setAttribute('aria-controls', 'main-menu');
    const updateMenu = () => menuToggle.setAttribute('aria-expanded', String(menuToggle.classList.contains('is-active')));
    updateMenu();
    new MutationObserver(updateMenu).observe(menuToggle, { attributes: true, attributeFilter: ['class'] });
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && menuToggle.classList.contains('is-active')) { menuToggle.click(); menuToggle.focus(); }
    });
}
document.querySelectorAll('.menu .current > a').forEach(link => link.setAttribute('aria-current', 'page'));
const main = document.querySelector<HTMLElement>('main');
if (main) {
    main.id = 'main-content';
    main.tabIndex = -1;
    const skip = document.createElement('a');
    skip.className = 'skip-link'; skip.href = '#main-content'; skip.textContent = '跳到正文';
    document.body.prepend(skip);
}
const topButton = document.createElement('button');
topButton.type = 'button'; topButton.className = 'back-to-top'; topButton.textContent = '↑';
topButton.setAttribute('aria-label', '回到顶部'); topButton.title = '回到顶部'; topButton.hidden = true;
document.body.append(topButton);
const updateTopButton = () => { topButton.hidden = window.scrollY < 600; };
window.addEventListener('scroll', updateTopButton, { passive: true });
updateTopButton();
topButton.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
    main?.focus({ preventScroll: true });
});
