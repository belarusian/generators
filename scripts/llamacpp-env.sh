#!/usr/bin/env bash
#
# Source this to configure the "llamacpp" family.
#
#   source scripts/llamacpp-env.sh
#
# For remote servers (e.g., Mac Studio at 10.106.1.184):
#   LLAMA_HOST=10.106.1.184 source scripts/llamacpp-env.sh

HOST="${LLAMA_HOST:-localhost}"

export LLAMACPP_SERVERS="coder=http://${HOST}:8080,oracle=http://${HOST}:8081,vision=http://${HOST}:8082,embed=http://${HOST}:8083"
export LLAMACPP_HOST="http://${HOST}:8080"
export COMPASS_FAMILY=llamacpp

echo "llamacpp family active"
echo "  LLAMACPP_SERVERS=${LLAMACPP_SERVERS}"
echo "  LLAMACPP_HOST=${LLAMACPP_HOST}"
echo "  COMPASS_FAMILY=${COMPASS_FAMILY}"
