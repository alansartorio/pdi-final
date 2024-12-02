PANDOC=pandoc
PANDOC_SLIDE_OPTIONS=-F mermaid-filter --mathjax -t revealjs -s --include-in-header=styles.html --slide-level=2 -V revealjs-url=reveal.js -V theme=blood
IMAGES_DIR=output
IMAGES=$(shell find $(IMAGES_DIR) -type f)
DIST_DIR=dist
DIST=$(DIST_DIR)/index.html $(DIST_DIR)/reveal.js $(IMAGES_DIR:%=$(DIST_DIR)/%)

informe: informe.pdf
slides: $(DIST)


informe.pdf: informe.md $(IMAGES)
	pandoc -o $@ --toc=true $<

dist/index.html: slides.md styles.html
	mkdir -p dist
	$(PANDOC) $(PANDOC_SLIDE_OPTIONS) -o $@ $<

dist/%: %
	mkdir -p $(dir $@)
	cp -r $< $(dir $@)

dist/reveal.js:
	wget https://github.com/hakimel/reveal.js/archive/master.tar.gz
	mkdir -p dist/reveal.js
	tar -xf master.tar.gz -C dist/reveal.js reveal.js-master/dist/ reveal.js-master/plugin/ --strip-components=1

clean-slides:
	$(RM) -rf dist

.PHONY: informe clean-slides