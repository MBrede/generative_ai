#!/usr/bin/env sh
# Start the next-token explorer. Endpoint URL/KEY come from example_solutions/session10/.env.
cd "$(dirname "$0")/.." || exit 1
exec uv run --env-file example_solutions/session10/.env \
  --with streamlit,streamlit-keyup,openai,umap-learn,plotly,pandas \
  streamlit run _src/prompt_focus_app.py "$@"
