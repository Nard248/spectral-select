# Makefile for ME-HSI Wavelength Selection Paper
# Compile with: make

MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib sections/*.tex
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc
	rm -f sections/*.aux

distclean: clean
	rm -f $(MAIN).pdf

.PHONY: all clean distclean
