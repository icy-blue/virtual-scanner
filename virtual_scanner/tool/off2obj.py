import os
import sys
if sys.platform == 'darwin':
    raise NotImplementedError('No pymeshlab plugins.')
import pymeshlab as m
import glob

class Off2Obj:
    # Function to convert an OFF file to OBJ format using PyMeshLab
    @staticmethod
    def convert_off_to_obj_with_pymeshlab(off_file, obj_folder):
        try:
            # Create a MeshSet
            ms = m.MeshSet()

            # Import the OFF file
            ms.load_new_mesh(off_file)

            # Create the output OBJ file path
            file_name = os.path.splitext(os.path.basename(off_file))[0]
            obj_file = os.path.join(obj_folder, f"{file_name}.obj")

            # Export the mesh as an OBJ file
            ms.save_current_mesh(obj_file)

            print(f"Converted {off_file} to {obj_file}")
        except Exception as e:
            print(f"Error converting {off_file}: {str(e)}")

    # Function to recursively find and convert OFF files
    @staticmethod
    def find_and_convert_off_files(input_folder, obj_folder):
        print(os.path.join(input_folder, "/**/*.off"))
        files = glob.glob(os.path.join(input_folder, "/**/*.off"), recursive=True)
        print(files)
        for file in files:
            Off2Obj.convert_off_to_obj_with_pymeshlab(file, obj_folder)

# print(123)
# find_and_convert_off_files('D:\\data\\meshes_data', 'D:\data\meshes_data\obj')


