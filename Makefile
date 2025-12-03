.PHONY: all clean

all: docs/main.pdf

docs/main.pdf: docs/main.typ
	typst compile $< $@

clean:
	rm -f docs/main.pdf
