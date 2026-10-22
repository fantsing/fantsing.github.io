/*!
*   Hugo Theme Stack
*
*   @author: Jimmy Cai
*   @website: https://jimmycai.com
*   @link: https://github.com/CaiJimmy/hugo-theme-stack
*/
import StackGallery from "ts/gallery";
import menu from 'ts/menu';
import createElement from 'ts/createElement';
import StackColorScheme from 'ts/colorScheme';
import { setupScrollspy } from 'ts/scrollspy';
import { setupSmoothAnchors } from "ts/smoothAnchors";

let Stack = {
    init: () => {
        /**
         * Bind menu event
         */
        menu();

        const articleContent = document.querySelector('.article-content') as HTMLElement;
        if (articleContent) {
            new StackGallery(articleContent);
            setupSmoothAnchors();
            setupScrollspy();
        }

        /**
         * Add copy button to code block
        */
        const highlights = document.querySelectorAll('.article-content div.highlight');

        highlights.forEach(highlight => {
            const codeBlock = highlight.querySelector('code');
            if (!codeBlock) return;
            const copyButton = document.createElement('button');
            copyButton.type = 'button';
            copyButton.textContent = '复制代码';
            copyButton.setAttribute('aria-label', '复制代码');
            copyButton.setAttribute('aria-live', 'polite');
            copyButton.classList.add('copyCodeButton');
            highlight.appendChild(copyButton);
            copyButton.addEventListener('click', async () => {
                try {
                    await navigator.clipboard.writeText(codeBlock.textContent || '');
                    copyButton.textContent = '已复制';
                } catch {
                    copyButton.textContent = '请选中代码复制';
                }
                setTimeout(() => { copyButton.textContent = '复制代码'; }, 1800);
            });
        });

        new StackColorScheme(document.getElementById('dark-mode-toggle'));
    }
}

window.addEventListener('DOMContentLoaded', () => {
    setTimeout(function () {
        Stack.init();
    }, 0);
})

declare global {
    interface Window {
        createElement: any;
        Stack: any
    }
}

window.Stack = Stack;
window.createElement = createElement;