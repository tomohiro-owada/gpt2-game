#!/bin/bash
# Deploy the GPT-2 game to cute.sus.cat/gpt2/ and verify it serves.
#
# Mirrors web/ -> the nginx web root (rsync --delete, so renamed/removed files
# are handled and no stale files linger), normalises ownership/perms, then
# does an HTTP serve-check through nginx — including every file referenced by
# models.json. No nginx config change is needed; the existing `location /`
# already serves this subdir.
#
# Needs root to write /var/www, so run with sudo:
#   sudo /home/dev/gpt2-game/deploy.sh
set -euo pipefail

SRC="/home/dev/gpt2-game/web/"
DST="/var/www/cute.sus.cat/gpt2"
HOST="cute.sus.cat"
BASE="http://127.0.0.1/gpt2"

if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR: must run as root to write $DST  ->  sudo $0" >&2
  exit 1
fi

echo "== Mirroring $SRC -> $DST/ =="
mkdir -p "$DST"
rsync -a --delete --chown=root:root --chmod=F644,D755 "$SRC" "$DST/"

echo "== Serve-check via nginx =="
fail=0
check() { # path
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' -H "Host: $HOST" "$BASE/$1")
  printf '  %-30s %s\n' "/${1:-}" "$code"
  [ "$code" = "200" ] || fail=1
}

# the page itself + every file that exists in the deployed dir
check ""
for f in $(cd "$DST" && ls -1); do check "$f"; done

# end-to-end: confirm each model file named in models.json resolves
if [ -f "$DST/models.json" ]; then
  echo "== models.json references =="
  python3 - "$DST/models.json" <<'PY'
import json,sys,subprocess
m=json.load(open(sys.argv[1]))
bad=0
for mod in m.get("models",[]):
    code=subprocess.run(["curl","-s","-o","/dev/null","-w","%{http_code}",
        "-H","Host: cute.sus.cat","http://127.0.0.1/gpt2/"+mod["file"]],
        capture_output=True,text=True).stdout
    print(f"  {mod['slug']:<18} -> {mod['file']:<28} {code}")
    if code!="200": bad=1
sys.exit(bad)
PY
  [ $? -eq 0 ] || fail=1
fi

if [ "$fail" -eq 0 ]; then
  echo "== OK: deployed and serving at https://$HOST/gpt2/ =="
else
  echo "== WARNING: one or more checks did not return 200 ==" >&2
  exit 1
fi
