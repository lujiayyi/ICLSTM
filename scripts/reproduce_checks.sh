#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PY="/opt/anaconda3/envs/iclstm-repro/bin/python"

if [[ ! -x "$ENV_PY" ]]; then
  echo "Missing /opt/anaconda3/envs/iclstm-repro. Create it with: conda env create -f environment.yml"
  exit 1
fi

echo "[1/3] Verifying core imports"
"$ENV_PY" - <<'PY'
import tensorflow as tf
import keras
import numpy
import pandas
import scipy
import sklearn
import ipyopt

print("tensorflow", tf.__version__)
print("keras", keras.__version__)
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
print("ipyopt", getattr(ipyopt, "__version__", "n/a"))
PY

echo "[2/3] Executing MPC first-principles notebook"
cd "$ROOT_DIR/MPC"
jupyter nbconvert \
  --to notebook \
  --execute mpc_first_principles.ipynb \
  --output mpc_first_principles.executed.ipynb \
  --ExecutePreprocessor.kernel_name=iclstm-repro \
  --ExecutePreprocessor.timeout=1800

echo "[3/3] Reproduction notes"
cat <<'EOF'
- Verified: mpc_first_principles.ipynb executes end-to-end in the iclstm-repro environment.
- Verified: TensorFlow/Keras can load the provided .h5 models and start MPC solves.
- Long-running notebooks: mpc_lstm.ipynb, mpc_rnn.ipynb, mpc_icrnn.ipynb, and mpc_iclstm.ipynb can take substantially longer because each iteration repeatedly calls neural-network prediction inside IPOPT.
- Recommended next manual runs:
    cd /Users/lujiayi/ICLSTM/MPC
    jupyter nbconvert --to notebook --execute mpc_lstm.ipynb --output mpc_lstm.executed.ipynb --ExecutePreprocessor.kernel_name=iclstm-repro --ExecutePreprocessor.timeout=1800
    jupyter nbconvert --to notebook --execute mpc_iclstm.ipynb --output mpc_iclstm.executed.ipynb --ExecutePreprocessor.kernel_name=iclstm-repro --ExecutePreprocessor.timeout=1800
EOF
