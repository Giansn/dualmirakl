.PHONY: setup start stop test audit models

setup:
	bash scripts/setup.sh

start:
	bash scripts/start_all.sh

stop:
	bash scripts/stop_all.sh

test:
	python -m pytest tests/ -v

audit:
	bash scripts/audit_env.sh

models:
	bash scripts/pull_models.sh
