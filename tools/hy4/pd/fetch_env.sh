#!/bin/bash
# fetch_env.sh: the dev node's environment at the same paths (the /opt Neuron venv, /source-repos, the code, small models)
S=http://172.31.47.244:8090/work/dist
set -e
curl -s $S/optvenv.tar | tar xf - -C /
curl -s $S/source_repos.tar | tar xf - -C /
curl -s $S/code.tar | tar xf - -C /mnt/nvme
curl -s $S/models_small.tar | tar xf - -C /mnt/nvme
id ssm-user >/dev/null 2>&1 || useradd -m -u 1001 -s /bin/bash ssm-user
echo ENV_DONE
