
targets := $(patsubst src/%.py,figures/%.ipynb,$(wildcard src/*.py))

requirements:
	pip install -r requirements.txt

all: $(targets)

format:
	black src

build/%.ipynb:
	echo $% $@
	mkdir -p build
	mkdir -p figures
	python -m jupytext src/$*.py --from py --to ipynb --output build/$*.ipynb

figures/%.ipynb: build/%.ipynb
	echo build/$*.ipynb $@
	python -m jupyter nbconvert --ExecutePreprocessor.kernel_name=python3 --to notebook --execute build/$*.ipynb --output $*_tmp.ipynb
	mv build/$*_tmp.ipynb $@

.PHONY: clean
