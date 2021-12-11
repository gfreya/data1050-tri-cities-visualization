#!/usr/bin/env bash
set -xe

mkdir -p /var/log;
chmod -R 777 /var/log;
mongod --fork --logpath=/var/log/mongodb.log;
python3 data_acquire.py & python3 app.py;
