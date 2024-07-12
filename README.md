# Install

```
python3 -m venv venv
source venv/bin/activate.yourshell
pip install -r requirements.frozen.txt
```

To update required packages, re-run: `pip freeze > requirements.frozen.txt`

# Dependency notes

`trimesh` requires `pyglet<2`. In order for closing a window to not terminate the entire program's execution on macOS, we further need `pyglet < 1.5.28`. This is all specified in `requirements.frozen.txt`.
