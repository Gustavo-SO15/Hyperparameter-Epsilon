echo "=== Acquiring datasets ==="
echo "---"

mkdir -p data
cd data

echo "- Downloading Tiny Imagenet"
wget --continue  http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -q tiny-imagenet-200.zip


echo "---"
echo "Happy classification Tiny Imagenet :)"
