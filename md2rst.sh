for md in $(find . -name '*.md'); do
    pandoc --from=markdown --to=rst --output=$(dirname $md)/$(basename).rst $md;
done