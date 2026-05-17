# Contributing to moonxi-net

Thanks for your interest in moonxi-net! This is a deep learning framework written in MoonBit with CUDA/cuDNN support. Contributions of all kinds are welcome: bug fixes, new layers, optimizer improvements, documentation, and examples.

## Development Setup

### Install MoonBit

This project requires the [CUDA-patched fork of MoonBit](https://github.com/moonxi-net/moon). Build and install it from that repository. The standard MoonBit toolchain won't work for GPU packages.

### Clone and build

```bash
git clone https://github.com/moonxi-net/moonxi-net.git
cd moonxi-net
moon update
moon check
```

### Install git hooks

The pre-commit hook runs `moon check` to catch type errors before they land in CI:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

## Code Style

- Run `moon fmt` before committing. Unformatted code won't pass review.
- MoonBit uses **block style**: separate blocks with `///|`. Each block is independent, so you can reorder or edit them without affecting the others.
- Functions use `snake_case`. Types use `PascalCase`.
- Keep files small and cohesive. A file should do one thing well.

## PR Checklist

Run these before opening a pull request:

```bash
moon check --deny-warn         # Type check (stricter than pre-commit hook)
moon fmt                       # Format
moon info && git diff --exit-code  # Verify interfaces unchanged (or changed intentionally)
moon test --target native      # Run tests
```

If `moon info` changes a `.mbti` file, check the diff. If it's intentional (new public API, changed signature), that's fine. If it's accidental, you may have exposed something you didn't mean to.

## Testing Guidelines

MoonBit has two kinds of test files:

- **Black-box tests** (`_test.mbt`): test the public API. Import the package with `@package.fn` and call exported functions.
- **White-box tests** (`_wbtest.mbt`): test internals. These have access to private members of the package.

For assertions:

- Use `assert_eq` and `assert_true` for deterministic, stable results.
- Use `inspect(value, content="...")` for snapshot tests. Update snapshots with `moon test --update`.

One gotcha: the MoonBit test runner captures stdout, so `println` output won't show up during tests. Use `inspect` or assertion functions instead.

### CPU vs GPU

CPU packages don't need CUDA:

```
nparray, grad, model, optimizer, train, loss, dataloader, utils,
datasets/cifar10, datasets/mnist
```

GPU packages require CUDA hardware and the CUDA toolkit:

```
gpu/cuda, gpu/tensor, gpu/event_tensor
```

CI only tests CPU packages. If you change GPU code, make sure to test locally with an NVIDIA GPU.

## Reporting Issues

Found a bug? Have a feature request? Open an issue at [GitHub Issues](https://github.com/moonxi-net/moonxi-net/issues).

Please include:

- MoonBit version (`moon version`)
- CUDA toolkit version (if GPU-related)
- Minimal reproduction steps
- Expected vs. actual behavior

## License

By contributing, you agree that your changes will be licensed under the [MIT License](LICENSE).
