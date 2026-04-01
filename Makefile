.PHONY: go setup start stop restart status test audit models

go:
	bash scripts/go.sh

setup:
	bash scripts/setup.sh

start:
	bash scripts/start_all.sh

stop:
	bash scripts/stop_all.sh

restart:
	bash scripts/go.sh --restart

status:
	bash scripts/status.sh

test:
	python -m pytest tests/ -v

audit:
	bash scripts/audit_env.sh

models:
	bash scripts/pull_models.sh
