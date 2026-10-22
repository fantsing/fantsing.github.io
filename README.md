# Lee Sivan's Blog

Hugo + Stack 主题的中文技术博客，发布于 <https://fantsing.github.io/>。

## 本地预览

项目使用 **Hugo Extended 0.151.0**。Windows 已附带 `hugo.exe`：

```powershell
.\hugo.exe server --disableFastRender
```

## 构建与检查

```powershell
.\hugo.exe --minify --gc --cleanDestinationDir --panicOnWarning --printPathWarnings
python scripts/check_site.py public
```

检查脚本只需要 Python 标准库，会检查站内链接、图片与样式资源、目录锚点、文章标题、搜索索引、RSS 和 sitemap。脚本会发现重复文章地址导致的页面覆盖。

## 写文章

在 `content/post/文章名称/index.md` 中编辑文章。建议使用稳定且唯一的 `slug`，避免改标题时网址变化：

```toml
+++
title = '文章标题'
slug = 'unique-post-slug'
date = '2026-10-22T10:00:00+08:00'
description = '一两句话说明文章内容。'
categories = ['计算机视觉']
tags = ['CNN']
+++
```

现有分类为计算机视觉、医学图像、科学机器学习、大模型与应用、算法与基础。

静态图片放在 `static/images/`，Markdown 中用 `/images/文件名.png` 引用，**不要写 `../../../static/`**。建议使用不含空格和百分号的文件名。文章目录中的图片也可以使用 Hugo page bundle 相对路径。

行内公式使用 `$...$` 或 `\(...\)`，独立公式使用 `$$...$$` 或 `\[...\]`。代码使用 Markdown 围栏，并注明语言。

## 发布

检查通过后，将需要发布的源文件提交并推送到 `main`。GitHub Actions 自动完成构建、站点检查与 GitHub Pages 发布。可以在仓库的 Actions 页面查看结果。

`public/`、`resources/`、`.audit/` 是生成或测试目录，不需要提交。自定义模板在 `layouts/`，样式在 `assets/scss/custom.scss`，交互修复在 `assets/ts/`，不直接修改上游主题。

公式和图片查看组件随站点托管在 `static/vendor/`，版本与来源记录在 `manifest.json`，第三方许可保留在对应目录。
