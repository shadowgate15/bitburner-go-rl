#!/usr/bin/env bash

set -e

exec python __main__.py $TRAIN_ARGS "$@"
