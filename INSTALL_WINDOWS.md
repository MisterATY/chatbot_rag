# Windows: Avoid Building llama-cpp-python

On Windows, **do not** install `llama-cpp-python` with a plain:

```bash
pip install llama-cpp-python
```

That tries to **build from source** and fails if you don't have Visual Studio Build Tools (nmake, C++ compiler). You don't need to install those.

## Option 1: You are NOT using GGUF (recommended if you use Qwen3-Embedding-4B from HuggingFace)

- In `app/config.py` leave **`QWEN_EMBEDDING_GGUF_PATH = ""`** (empty).
- **Do not install** `llama-cpp-python`. The app works without it.
- Use **Qwen/Qwen3-Embedding-4B** (or 0.6B/8B) from HuggingFace; no GGUF needed.

## Option 2: You ARE using a GGUF embedding model

Install **pre-built wheels** so pip does **not** build from source:

**CPU only:**

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

**With NVIDIA GPU (CUDA 12.x):**

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

(Other CUDA versions: `cu118`, `cu124`, etc. – see [llama-cpp-python docs](https://github.com/abetlen/llama-cpp-python#installation-with-hardware-acceleration).)

The `--extra-index-url` makes pip **download a ready-made wheel** instead of compiling. No Visual Studio or C++ tools needed on your PC.

## If you really want to build from source

You would need to install **Visual Studio Build Tools** with the **“Desktop development with C++”** workload (includes nmake and MSVC). That is only necessary if pre-built wheels are not an option.
