# Poster fonts

EB Garamond and Open Sans, the primary typefaces in Duke's brand guide (brand.duke.edu/typography).
Both are open-licensed under the SIL Open Font License 1.1 and come from Google Fonts.

`static/` holds static weights (400, 600, 700, plus 400 italic; Latin and Greek subsets) cut from the
variable fonts, so Chrome embeds them as TrueType in the poster PDFs. `fonts.css` is the `@font-face`
list that `poster/build_poster.py` inlines.

If this repository is ever made public, add the OFL license text for each family alongside these files.
