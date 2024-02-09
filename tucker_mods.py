# import glob
# import os
def delete_files_by_extension(extension):
    from glob import glob
    from os import remove
    files_to_delete = glob(f"*results/*.{extension}")
    for file_path in files_to_delete:
        remove(file_path)
        print(f"File '{file_path}' deleted.")

extension_to_delete = "png"  # Change this to the extension you want to delete
delete_files_by_extension(extension_to_delete)
extension_to_delete = "xdmf"  # Change this to the extension you want to delete
delete_files_by_extension(extension_to_delete)
extension_to_delete = "h5"  # Change this to the extension you want to delete
delete_files_by_extension(extension_to_delete)

def print_variable(variable):
    from numpy import size
    variable_name = [name for name, value in globals().items() if value is variable] #[0]
    print(f"Variable name using globals(): {variable_name}")
    print('type: ',str(type(variable)))
    print('size: ',str(size(variable)))
    print('value = ',str(variable))
    
def check_clock(): 
    from time import time
    return time()