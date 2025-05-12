# Wes Lee's Portfolio Website

This is the source code for my personal portfolio website, built with Jekyll and the Hydejack theme. The site showcases my projects, skills, and professional experience in AI engineering, data science, and machine learning.

## Site Structure

- **About**: Professional background, experience, education, and skills
- **Projects**: Portfolio of data science and AI projects
- **Skills**: Detailed breakdown of technical competencies
- **Blog**: Articles and insights (coming soon)
- **Contact**: Ways to get in touch

## Technical Setup

This site uses:
- Jekyll static site generator
- Hydejack theme (v9.1.6)
- GitHub Pages for hosting

## Local Development

To run this site locally:

1. Make sure you have Ruby and Bundler installed
2. Clone this repository
3. Run `bundle install` to install dependencies
4. Run `bundle exec jekyll serve` to start the development server
5. Visit `http://localhost:4000` in your browser

## Deployment

This site is automatically deployed via GitHub Pages whenever changes are pushed to the main branch.

To deploy manually:

1. Push changes to the main branch
2. GitHub Actions will build and deploy the site
3. Visit `https://adredes-weslee.github.io` to see the live site

## Content Updates

- **Adding a new project**: Create a new markdown file in the `_projects` folder
- **Writing a blog post**: Add a markdown file to the `_posts` folder with the format `YYYY-MM-DD-title.md`
- **Updating skills**: Modify the `skills.md` file

## Future Improvements

- Add actual project images instead of placeholders
- Implement blog functionality
- Add project filtering
- Create a PDF resume download option
- Add dark mode toggle

## License

© 2025 Wes Lee. All rights reserved.