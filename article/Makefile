LATEX=pdflatex
LATEXOPT=--shell-escape

LATEXMK=latexmk
LATEXMKOPT=-pdf

MAIN = ontherun
REVISED = revised
SOURCES = $(MAIN).tex Makefile journal_defs.tex
REVISED_SOURCES = $(REVISED).tex Makefile journal_defs.tex 
FIGURES := $(shell find plots/* -type f)

all:    $(MAIN).pdf $(REVISED).pdf

.refresh:
	touch .refresh

$(MAIN).pdf: $(MAIN).tex .refresh $(SOURCES) $(FIGURES)
		$(LATEXMK) $(LATEXMKOPT) $(MAIN)

$(REVISED).pdf: $(REVISED).tex .refresh $(REVISED_SOURCES) $(FIGURES)
		$(LATEXMK) $(LATEXMKOPT) $(REVISED)

force:
		touch .refresh
		rm $(MAIN).pdf
		$(LATEXMK) $(LATEXMKOPT) $(MAIN)

clean:
		$(LATEXMK) -C $(MAIN)
		$(LATEXMK) -C $(REVISED)
		rm -rf *~ *.tmp
		rm -f *.bbl *.blg *.aux *.end *.fls *.log *.out *.fdb_latexmk

once:
		$(LATEXMK) $(LATEXMKOPT) $(MAIN)

debug:
		$(LATEX) $(LATEXOPT) $(MAIN)

.PHONY: clean force once all