<!-- SYMONEURAL_BOOTSTRAP_START -->
## First-time setup: required before audits or server start

Do **not** run `production_audit.py`, `scripts/start.sh`, or `scripts/test.sh` before the Python environment exists and the repo requirements are installed.

Correct first-time setup from the repo root:

```bash
cd ~/SyMoNeuRaL_MoE
chmod +x scripts/bootstrap_env.sh scripts/start.sh scripts/stop.sh scripts/status.sh scripts/test.sh
scripts/bootstrap_env.sh
```

What `scripts/bootstrap_env.sh` does:

```text
1. verifies Python is available
2. creates or reuses .venv
3. activates .venv
4. upgrades pip, setuptools, and wheel
5. installs requirements.txt
6. runs pip check
7. verifies app imports
8. runs production_audit.py
```

After bootstrap succeeds:

```bash
source .venv/bin/activate
python3 production_audit.py --live
scripts/start.sh
scripts/test.sh
```

<!-- SYMONEURAL_BOOTSTRAP_END -->


# Bootstrap before audits

The production audit must be run from the repo virtual environment. A clean machine setup is:

```bash
cd ~/SyMoNeuRaL_MoE
chmod +x scripts/bootstrap_env.sh
scripts/bootstrap_env.sh
```

Only after that should these be run manually:

```bash
source .venv/bin/activate
python3 production_audit.py
python3 production_audit.py --live
```

---

# Production Runbook

## Install

```bash
cp config/symoneural.env.example .env
scripts/install_cuda_vllm.sh
```

## Edit config

```bash
nano .env
```

Set model paths.

## Audit

```bash
. .venv/bin/activate
python3 production_audit.py --static-only
python3 production_audit.py
python3 production_audit.py --live
```

## Start/stop/status

```bash
scripts/start.sh
scripts/status.sh
scripts/test.sh
scripts/stop.sh
```

## Logs

```bash
tail -f data/runtime/symon-manager.log
```

## Health

```bash
curl http://127.0.0.1:8000/health | jq
curl http://127.0.0.1:8000/ready | jq
curl http://127.0.0.1:8000/metrics
```

`/health` may be degraded for model setup. `/ready` should be 200 only when production worker readiness is true.

## Deployment with systemd

See `deploy/systemd/symoneural.service`.
