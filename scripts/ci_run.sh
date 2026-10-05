#!/usr/bin/env bash
# One relay leg: restore state, run the fleet for ~5h40m, saving state to the `state` branch every 5 min.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
END=$(( $(date +%s) + ${RUN_MINUTES:-340} * 60 ))
REPO_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"

restore() {
  mkdir -p runs; rm -rf /tmp/state
  if git ls-remote --exit-code --heads "$REPO_URL" state >/dev/null 2>&1; then
    git clone -q --depth 1 --branch state "$REPO_URL" /tmp/state && cp -R /tmp/state/runs/. runs/ 2>/dev/null
    echo "restored state: $(ls runs | wc -l) entries"
  else echo "no state branch yet"; fi
}

save() {
  local s=/tmp/save; rm -rf $s; mkdir -p $s/runs
  for db in runs/*/*.sqlite; do
    [ -f "$db" ] || continue
    mkdir -p "$s/$(dirname "$db")"; sqlite3 "$db" ".backup '$s/$db'" 2>/dev/null
  done
  [ -f runs/benchmark.json ] && cp runs/benchmark.json $s/runs/
  for r in runs/*/retired; do [ -f "$r" ] && { mkdir -p "$s/$(dirname "$r")"; touch "$s/$r"; }; done
  { echo "# Fleet status"; echo "updated $(date -u '+%Y-%m-%d %H:%M UTC')"; echo '```'
    .venv/bin/python scripts/fleet.py status 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | head -n 120; echo '```'; } > $s/STATUS.md
  RUNS_DIR=$s/runs .venv/bin/python scripts/export_json.py $s/data.json 2>&1 | tail -n 3
  ( cd $s && git init -q -b state && git config user.email ci@example.com && git config user.name fleet-ci \
    && git add -A && git commit -q -m "state $(date -u +%FT%TZ)" && git push -q -f "$REPO_URL" state ) || echo "state push failed"
}

prune() {  # drop run folders of strategies that were removed from the repo
  for d in runs/*/; do n=$(basename "$d")
    case "$n" in portfolio|daytrader|multi|dashboard) continue;; esac
    [ -f "user_data/strategies/$n.py" ] || { rm -rf "$d"; echo "pruned $n"; }
  done
}

restore
prune
.venv/bin/python scripts/fleet.py start all
echo "started: $(pgrep -fc 'freqtrade trade') bots"
while [ "$(date +%s)" -lt "$END" ]; do
  sleep 300
  # restart anything that died (idempotent)
  .venv/bin/python scripts/fleet.py start all >/dev/null 2>&1
  save
done
.venv/bin/python scripts/fleet.py stop
sleep 15
save
