#!/bin/bash
# pd.sh <prefill|decode|both> '<commands>' [timeout_s]: run a shell snippet on the PD nodes over SSM (as root, in
# /mnt/nvme), print each node's output
R=ap-south-2; declare -A ID=([prefill]=i-03a3d5f11d86e5db3 [decode]=i-0aa05f3deb2f888f1)
nodes=$1; [ "$1" = both ] && nodes="prefill decode"
cmd=$(python3 -c 'import base64,json,sys; b=base64.b64encode(("cd /mnt/nvme 2>/dev/null; "+sys.argv[1]).encode()).decode(); print(json.dumps({"commands":["echo "+b+" | base64 -d | bash -l"],"executionTimeout":[sys.argv[2]]}))' "$2" "${3:-3600}")
declare -A CID
for n in $nodes; do CID[$n]=$(aws ssm send-command --region $R --instance-ids ${ID[$n]} --document-name AWS-RunShellScript --parameters "$cmd" --query Command.CommandId --output text); done
for n in $nodes; do
  while true; do st=$(aws ssm get-command-invocation --region $R --instance-id ${ID[$n]} --command-id ${CID[$n]} --query Status --output text 2>/dev/null); case $st in Success|Failed|Cancelled|TimedOut) break;; esac; sleep 3; done
  echo "=== $n [$st]"; aws ssm get-command-invocation --region $R --instance-id ${ID[$n]} --command-id ${CID[$n]} --query '[StandardOutputContent,StandardErrorContent]' --output text
done
