# IMPACT (Intention-Modulated Personalized ACT-R)

## Development

### Pre-commit Hooks

[prek](https://prek.j178.dev/) (which is a "reimagined version of [pre-commit](https://pre-commit.com/), written in Rust") runs automated checks before every `git commit`.

**Setup:**

```bash
uv tool install prek  # install prek (recommended way - using `uv tool` command)
prek install          # install hooks into .git\hooks\pre-commit
```

Now every `git commit` will invoke `prek run` for the files **included in the commit**. If you ever want to undo this, run `prek uninstall`.

If you want to manually execute all configured hooks on the files in your current git staging area, run:

```bash
prek run
```

To target all files (the entire repository), add `--all-files` flag to the command:

```bash
prek run --all-files
```

To bypass all hooks, add `--no-verify` flag to the `git commit` command:

```bash
git commit --no-verify -m "your message"
```
