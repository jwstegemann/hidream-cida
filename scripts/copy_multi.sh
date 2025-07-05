modal volume ls dataset-vgg-casia /training/real --json | jq -r '.[].Filename' | while read -r fname; do
  echo "Processing $fname..."
  modal volume cp dataset-vgg-casia "$fname" / --recursive
  modal volume rm dataset-vgg-casia "$fname" --recursive
done
