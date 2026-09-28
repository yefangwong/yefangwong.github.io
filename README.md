# Ye-Fang Wong Personal Technical Blog & R&D Hub

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)](https://yefangwong.github.io)
[![Jekyll](https://img.shields.io/badge/Jekyll-v3.8.5-blue)](https://jekyllrb.com/)
[![Google Tag Manager](https://img.shields.io/badge/GTM-GTM--NKTF823T-orange)](https://tagmanager.google.com/)

This repository hosts the official personal website and technical log of **Ye-Fang Wong (翁藝芳)**, Senior Software Developer with 25+ years of experience in Search Engine Engineering, High-Concurrency Backend Systems, and Responsible AI Guardrails.

- **Live URL**: [https://yefangwong.github.io](https://yefangwong.github.io)
- **Primary Domain**: Responsible AI, Neuro-Symbolic Knowledge Grounding, Sequence Logic (IRNN), and Mission-Critical Architecture.

---

## 📁 Repository Directory Structure

```text
.
├── index.html              # Homepage (Hero section, Featured R&D, Technical Logs index)
├── about/
│   └── index.html          # "About My Technical Journey" (Career evolution & R&D milestones)
├── _layouts/
│   └── post.html           # Universal article layout (MathJax 3, GTM, responsive CSS, typography)
├── _posts/                 # Technical Markdown posts (YYYY-MM-DD-title.md)
│   ├── 2026-09-27-grounding-dense-embeddings-via-hownet-retrofitting.md    # Traditional Chinese version
│   └── 2026-09-27-grounding-dense-embeddings-via-hownet-retrofitting-en.md # English version
├── assets/
│   └── main.css            # Site global styles & Rouge syntax highlighter theme
├── _config.yml             # Jekyll site configuration
├── 404.html                # Custom 404 error page
└── README.md               # Site maintenance documentation
```

---

## 🛠️ Site Maintenance & Content Management

### 1. Adding a New Blog Post

To publish a new technical article, create a Markdown file in `_posts/` adhering to the naming convention `YYYY-MM-DD-your-post-title.md`.

#### Standard Front Matter Header:
```markdown
---
layout: post
title: "Your Post Title Here"
date: YYYY-MM-DD HH:MM:SS +0800
categories: [AI, NLP, Architecture]
tags: [Retrofitting, HowNet, Responsible-AI]
author: "Ye-Fang Wong (翁藝芳)"
lang: zh
---

> 🌐 **Language / 語言切換**: **繁體中文 (Current)** • [English Version →](/posts/YYYY-MM-DD-title-en/)

Your content goes here...
```

### 2. Embedded Math & Formula Support (MathJax 3)

Math formulas are natively rendered via MathJax 3 configured in `_layouts/post.html`:
- **Inline math**: `$E = mc^2$` or `\(p = 0.000080\)`
- **Display math block**:
  ```markdown
  $$
  \mathbf{q}_i^{(t+1)} = \frac{\alpha_i \hat{\mathbf{q}}_i + \sum_{j \in N(i)} \beta_{ij} \mathbf{q}_j^{(t)}}{\alpha_i + \sum_{j \in N(i)} \beta_{ij}}
  $$
  ```

### 3. Analytics & Google Tag Manager (GTM)

- **Container ID**: `GTM-NKTF823T`
- Integrated across all page templates (`index.html`, `about/index.html`, `_layouts/post.html`, `404.html`).
- To connect Google Analytics 4 (GA4):
  1. Open [Google Tag Manager Console](https://tagmanager.google.com/).
  2. Create a new tag of type **Google Tag / GA4 Tag**.
  3. Enter your GA4 Measurement ID (`G-XXXXXXXXXX`).
  4. Set trigger to **Initialization - All Pages**, and click **Publish**.

---

## 🚀 Local Development & Deployment

### Local Server Setup (Optional)

1. Ensure Ruby and Bundler are installed.
2. Install Jekyll dependencies:
   ```bash
   bundle install
   ```
3. Run local dev server:
   ```bash
   bundle exec jekyll serve
   ```
4. Access preview at `http://localhost:4000`.

### Production Deployment

Publishing is fully automated via GitHub Pages:
```bash
git add .
git commit -m "feat: publish new article on Neuro-Symbolic AI"
git push origin master
```
- Pushing to the `master` branch automatically triggers GitHub Pages build & deployment (typically live within 1–2 minutes).

---

## 📄 License & Attribution

- Content & Technical Articles © Ye-Fang Wong. All rights reserved.
