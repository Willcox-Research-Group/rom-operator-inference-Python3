.PHONY: all clean

%.pdf: %.tex config.tex
	latexmk -pdf -g $<
	pdfcrop $@ $@
	pdf2svg $@ $*/eq%02d.svg all
	# mv -v eq*.svg $*/

all: details.pdf documentation.pdf notation.pdf

clean:
	latexmk -c
	rm *.pdf

