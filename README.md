# cora_node_classification
conda create -n node_cora python=3.11
conda activate node_cora
chmod +x run.sh
docker build -t foo . && docker run -it foo
docker-compose up