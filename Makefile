.PHONY: all clean

all: report/main.pdf

report/main.pdf: report/main.typ
	typst compile $< $@

clean:
	rm -f report/main.pdf
