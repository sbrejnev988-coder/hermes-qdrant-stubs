#!/bin/bash
# Автостарт knowledge base серверов
# Добавить в crontab: @reboot bash /root/.hermes/knowledge_db/start_servers.sh

cd /root/.hermes/knowledge_db

# Убиваем старые если есть
pkill -f "qdrant_stub.py" 2>/dev/null
pkill -f "embed_stub.py" 2>/dev/null
sleep 1

# Запускаем (без http_proxy!)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
nohup python3 qdrant_stub.py > /dev/null 2>&1 &
nohup python3 embed_stub.py > /dev/null 2>&1 &

echo "Knowledge base servers started (qdrant:6333, embed:4000)"
