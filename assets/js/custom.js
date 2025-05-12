// Font Awesome Integration for Wes Lee's Portfolio

// Add Font Awesome script to the head
document.addEventListener('DOMContentLoaded', function() {
  const fontAwesomeScript = document.createElement('script');
  fontAwesomeScript.src = 'https://kit.fontawesome.com/a076d05399.js';
  fontAwesomeScript.crossOrigin = 'anonymous';
  document.head.appendChild(fontAwesomeScript);
  
  // Add icon to skills sections
  addSkillIcons();
  
  // Add icons to contact methods
  addContactIcons();
  
  // Add animation to project cards
  animateProjectCards();
});

// Add appropriate icons to skill categories
function addSkillIcons() {
  const skillSections = document.querySelectorAll('.skill-section h3');
  
  if (skillSections) {
    skillSections.forEach(section => {
      const text = section.textContent.toLowerCase();
      
      if (text.includes('machine learning') || text.includes('ai')) {
        addIcon(section, 'fas fa-brain');
      } else if (text.includes('programming') || text.includes('development')) {
        addIcon(section, 'fas fa-code');
      } else if (text.includes('devops') || text.includes('cloud')) {
        addIcon(section, 'fas fa-cloud');
      } else if (text.includes('data')) {
        addIcon(section, 'fas fa-database');
      }
    });
  }
}

// Add icons to contact methods
function addContactIcons() {
  const contactMethods = document.querySelectorAll('.contact-method h3');
  
  if (contactMethods) {
    contactMethods.forEach(method => {
      const text = method.textContent.toLowerCase();
      
      if (text.includes('email')) {
        addIcon(method, 'far fa-envelope');
      } else if (text.includes('phone')) {
        addIcon(method, 'fas fa-phone');
      } else if (text.includes('linkedin')) {
        addIcon(method, 'fab fa-linkedin');
      } else if (text.includes('github')) {
        addIcon(method, 'fab fa-github');
      }
    });
  }
}

// Add subtle animation to project cards
function animateProjectCards() {
  const projectCards = document.querySelectorAll('.project-card');
  
  if (projectCards) {
    projectCards.forEach(card => {
      card.addEventListener('mouseenter', () => {
        card.style.transform = 'translateY(-5px)';
        card.style.boxShadow = '0 10px 20px rgba(0,0,0,0.1)';
      });
      
      card.addEventListener('mouseleave', () => {
        card.style.transform = 'translateY(0)';
        card.style.boxShadow = '0 2px 5px rgba(0,0,0,0.05)';
      });
    });
  }
}

// Helper function to add icon to element
function addIcon(element, iconClass) {
  const icon = document.createElement('i');
  iconClass.split(' ').forEach(cls => icon.classList.add(cls));
  icon.style.marginRight = '0.5rem';
  element.prepend(icon);
}
