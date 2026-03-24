document.addEventListener('DOMContentLoaded', () => {
  initializeHomeLanding();
  observeRevealElements();
});

function initializeHomeLanding() {
  const homeShell = document.querySelector('.home-shell');

  if (!homeShell) {
    document.documentElement.classList.remove('home-landing-page');
    return;
  }

  document.documentElement.classList.add('home-landing-page');

  requestAnimationFrame(() => {
    homeShell.classList.add('is-ready');
  });
}

function observeRevealElements() {
  const revealNodes = document.querySelectorAll('[data-reveal]');

  if (!revealNodes.length || !('IntersectionObserver' in window)) {
    revealNodes.forEach((node) => node.classList.add('is-visible'));
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, {
    rootMargin: '0px 0px -12% 0px',
    threshold: 0.15,
  });

  revealNodes.forEach((node) => observer.observe(node));
}
