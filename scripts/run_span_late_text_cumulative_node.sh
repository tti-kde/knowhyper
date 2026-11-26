#!/usr/bin/env bash
set -uo pipefail

# Fixed cumulative NER experiment: #220 retrieved entity text in sequence B, followed by
# #231's frozen-inner ev_warm late-logit verifier. Keep this as a thin wrapper so every
# dispatch, provenance and circuit-breaker fix in the completed #231 driver is inherited.
# There are no edge/EV architecture or hyperparameter sweeps in this campaign.

export SPAN_LATE_CAMPAIGN=cumulative
exec bash "$(dirname "$0")/run_span_late_logit_node.sh" "$@"
