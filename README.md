# nlp-docs

nlp 自然语言处理文档

## 转移到 sphinx-doc

```sh
for md in $(find . -name '*.md'); do
    pandoc --from=markdown --to=rst --output=$(dirname $md)/$(basename $md).rst $md;
done
```
