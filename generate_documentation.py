import pdoc
import os
import glob

context = pdoc.Context()
module = pdoc.Module('muvi', context=context)

with open(os.path.join('docs', 'muvi.md'), 'wt') as f:
    f.write(module.text())
