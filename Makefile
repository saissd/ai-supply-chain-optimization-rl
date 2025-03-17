# Install dependencies
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

# Run tests with coverage
test:
	python -m pytest -vv --cov=. --cov-report=term-missing test.py

# Format Python files using Black
format:
	black *.py

# Run linter (ignoring test files)
lint:
	pylint --disable=R,C --ignore-patterns="test_.*?py" *.py 

# Lint Dockerfile (if applicable)
container-lint:
	@if [ -f Dockerfile ]; then docker run --rm -i hadolint/hadolint < Dockerfile; fi

# Run formatting and linting together
refactor: format lint

# Placeholder for deployment step
deploy:
	@echo "Deploy not implemented"

# Run all steps
all: install lint test format deploy
