#!/usr/bin/env bash
# Summarises AWS spending by service, to see where credits went and what is
# using money now. Run in AWS CloudShell:  ./aws_costs.sh
# Optional: START (default 2026-03-31), the first day to include.
set -euo pipefail
export AWS_PAGER=""

START=${START:-2026-03-31}
TODAY=$(date -u +%Y-%m-%d)
MONTH_START=$(date -u +%Y-%m-01)

summarise() {  # reads Cost Explorer JSON, prints rows >= $0.50 and totals by service
  python3 -c '
import json, sys
from collections import defaultdict
tot = defaultdict(float)
for r in json.load(sys.stdin)["ResultsByTime"]:
    m = r["TimePeriod"]["Start"][:7]
    for g in r["Groups"]:
        s = g["Keys"][0]
        a = abs(float(g["Metrics"]["UnblendedCost"]["Amount"]))
        tot[s] += a
        if a >= 0.5 and sys.argv[1] == "monthly":
            print("  %s  %-45s $%9.2f" % (m, s, a))
print("  Total by service:")
for s, a in sorted(tot.items(), key=lambda x: -x[1]):
    if a >= 0.5:
        print("    %-45s $%9.2f" % (s, a))
print("    %-45s $%9.2f" % ("ALL SERVICES", sum(tot.values())))
' "$1"
}

echo "=== 1. Credits applied, by service and month ($START to $TODAY) ==="
aws ce get-cost-and-usage --time-period Start="$START",End="$TODAY" \
  --granularity MONTHLY --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=SERVICE \
  --filter '{"Dimensions":{"Key":"RECORD_TYPE","Values":["Credit"]}}' \
  --output json | summarise monthly

echo
echo "=== 2. Usage this month so far ($MONTH_START to $TODAY), before credits ==="
if [ "$MONTH_START" = "$TODAY" ]; then
  echo "  (first day of the month: no data yet)"
else
  aws ce get-cost-and-usage --time-period Start="$MONTH_START",End="$TODAY" \
    --granularity MONTHLY --metrics UnblendedCost \
    --group-by Type=DIMENSION,Key=SERVICE \
    --filter '{"Not":{"Dimensions":{"Key":"RECORD_TYPE","Values":["Credit","Refund"]}}}' \
    --output json | summarise total
fi
