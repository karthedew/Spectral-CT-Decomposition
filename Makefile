.PHONY: all clean latex

all: latex

latex: 
	echo "Building PDF via LaTeX and BibTeX..."
	cd doc; \
	pdflatex Spectral_CT_Decomposition.tex; \
	bibtex Spectral_CT_Decomposition.aux; \
	pdflatex Spectral_CT_Decomposition.tex

clean: 
	echo "Cleaning auxiliary files..."
	cd doc; \
	rm -f *.aux *.bbl *.blg *.log *.out *.toc
