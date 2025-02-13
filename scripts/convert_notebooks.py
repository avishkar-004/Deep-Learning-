"""Convert Jupyter notebooks to Python scripts."""

import os
import json


def notebook_to_script(notebook_path, output_path=None):
    """Extract code cells from notebook to Python script."""
    if output_path is None:
        output_path = notebook_path.replace('.ipynb', '.py')
    
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            code_cells.append(source)
    
    with open(output_path, 'w') as f:
        f.write('# Auto-generated from ' + os.path.basename(notebook_path) + '\n\n')
        for i, cell in enumerate(code_cells):
            f.write(f'# Cell {i + 1}\n')
            f.write(cell + '\n\n')
    
    print(f"Converted: {notebook_path} -> {output_path}")


if __name__ == "__main__":
    assignments_dir = os.path.join(os.path.dirname(__file__), '..', 'Assignments')
    for f in sorted(os.listdir(assignments_dir)):
        if f.endswith('.ipynb'):
            notebook_to_script(os.path.join(assignments_dir, f))
