"""Copy already verified local JAX packages into a separate venv, without changing the source."""
from importlib import metadata
from pathlib import Path
import shutil
import sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

source = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
if source == target or not source.is_dir() or not target.is_dir():
    raise ValueError('Two distinct existing site-packages directories are required')
distributions = {canonicalize_name(d.metadata['Name']): d for d in metadata.distributions(path=[str(source)])}
roots = ['jax', 'jaxlib', 'jax-cuda13-plugin', 'jax-cuda13-pjrt', 'flax', 'chex',
         'optax', 'distrax', 'tfp-nightly', 'numpy', 'rich']
roots += [name for name in distributions if name.startswith('nvidia-')]
pending, selected = list(roots), set()
while pending:
    name = canonicalize_name(pending.pop())
    if name in selected or name not in distributions:
        continue
    selected.add(name)
    for spec in distributions[name].requires or []:
        requirement = Requirement(spec)
        if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
            pending.append(requirement.name)
for name in sorted(selected):
    dist = distributions[name]
    for item in dist.files or []:
        original = (source / item).resolve()
        if not original.is_relative_to(source) or not original.is_file() or original.suffix == '.pyc':
            continue
        dest = target / original.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, dest)
        shutil.copymode(original, dest)
    print(name, dist.version, flush=True)
