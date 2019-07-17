#!/usr/bin/env python
import glob
import os

if __name__ == '__main__':
    print('Searching for notebooks in the notebooks directory')
    # maybe executing like execute_notebook.py
    notebook_dir = '../notebooks'
    result_dir = '../results'
    if not os.path.exists(notebook_dir):
        # maybe executing like scripts/execute_notebook.py
        notebook_dir = './notebooks'
        result_dir = './results'
        if not os.path.exists(notebook_dir):
            # still did not find the notebook directory
            print('Notebook Directory not found! Exiting')
            exit(0)
    # glob notebooks
    notebooks = glob.glob(f'{notebook_dir}/*.ipynb')
    # the length cannot be 0
    if len(notebooks) == 0:
        print('No Notebooks found! Exiting.')
        exit(0)
    print('Select a notebook to run. Results will be logged to <notebook_name>.log in the results directory\n')
    for i in range(len(notebooks)):
        print(f'{i + 1}. {os.path.basename(notebooks[i])}')

    try:
        option = int(input('\nEnter option: '))
        if option > len(notebooks):
            assert IndexError
        print(f'Executing notebook {os.path.basename(notebooks[option - 1])}')
        # deal with spaces in file names
        selected_notebook = notebooks[option - 1].replace(' ', '\ ')
        result_file_name = os.path.splitext(os.path.basename(selected_notebook))[0]
        # run the selected notebook
        os.system(f'jupyter nbconvert --to script --execute --stdout {selected_notebook} | '
                  f'python -u 2>&1 | tee  {result_dir}/{result_file_name}.log &')
        print('Process started!')
    except IndexError as e:
        print('Invalid option! Existing.')
        exit(0)
