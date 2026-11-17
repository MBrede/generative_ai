#!/usr/bin/env sh
# Start the chunking-strategies demo. Endpoint URL/KEY come from example_solutions/session10/.env.
cd "$(dirname "$0")/.." || exit 1
exec uv run --env-file example_solutions/session10/.env \
  --with streamlit,openai,scikit-learn,plotly,pandas \
  streamlit run _src/chunking_app.py "$@"
