# Physical Environment

Our project is a physical environment to train LLMs to generate STL files, the same files used in physical CAD designs.

## Setup

```sh
$ pip install pyrender trimesh pyglet matplotlib torch transformers pydantic vllm numpy requests tenacity wandb
```

Shared libraries for Ubuntu GL rendering.
```sh
$ sudo apt-get install libglfw3-dev libgles2-mesa-dev libnvidia-gl-570-server
```