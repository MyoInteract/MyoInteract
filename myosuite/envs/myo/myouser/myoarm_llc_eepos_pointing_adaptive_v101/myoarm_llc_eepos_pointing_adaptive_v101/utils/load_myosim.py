import mujoco
import shutil
import os
import pathlib
import argparse

def load_myosim_model(myo_model_path):
    myo_model_path_filename = os.path.basename(myo_model_path)
    myo_model_path_subdirname = os.path.basename(os.path.dirname(myo_model_path))
    myo_model_path_complete = os.path.join(os.path.dirname(__file__), "../bm_models/_external_models/myo_sim/", myo_model_path_subdirname, myo_model_path_filename)

    model = mujoco.MjModel.from_xml_path(myo_model_path_complete)

    ## copy .xml file
    uitb_model_path = os.path.join(os.path.dirname(__file__), f"../bm_models/myo_{myo_model_path_subdirname}/", myo_model_path_filename)
    # pathlib.Path(os.path.dirname(uitb_model_path)).mkdir(parents=True, exist_ok=True)

    pathlib.Path(os.path.dirname(uitb_model_path)).mkdir(parents=True, exist_ok=True)   
    pathlib.Path(uitb_model_path).touch()
    mujoco.mj_saveLastXML(uitb_model_path, model)
    ## copy meshes and scene dirs
    uitb_model_assets_path = os.path.join(os.path.dirname(uitb_model_path), "assets")
    meshes_dir = os.path.join(os.path.dirname(myo_model_path_complete), "../meshes/")
    scene_dir = os.path.join(os.path.dirname(myo_model_path_complete), "../scene/")
    shutil.copytree(meshes_dir, uitb_model_assets_path,
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.pyc'))
    shutil.copytree(scene_dir, uitb_model_assets_path,
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.pyc'))

    ## modify relative paths in copied .xml file to refer to correct meshes/scene dir
    adjust_myosim_model_paths(uitb_model_path)
    print(f"CREATED: '{uitb_model_path}'")
    print(f"ADDED: relevant mesh and scene files were added to '{uitb_model_assets_path}'")

def adjust_myosim_model_paths(uitb_model_path):
    with open(uitb_model_path, 'r') as file:
        filedata = file.read()
    
    ## adjust meshdir
    filedata = filedata.replace("../myo_sim/meshes/", "./assets/").replace('meshdir="../"', 'meshdir="./"')

    ## adjust textdir
    filedata = filedata.replace("../myo_sim/scene/", "./assets/").replace('texturedir="../"', 'texturedir="./"')

    with open(uitb_model_path, 'w') as file:
        file.write(filedata)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Integrate all myosim models into the UitB bm_models structure.')
    parser.add_argument('myo_file', type=str,
                            help='the main myosim model file')
    args = parser.parse_args()

    load_myosim_model(args.myo_file)
