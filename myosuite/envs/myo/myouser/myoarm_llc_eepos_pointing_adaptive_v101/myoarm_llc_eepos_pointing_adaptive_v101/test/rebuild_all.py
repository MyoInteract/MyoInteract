import os
import uitb
from uitb.utils.functions import output_path, parse_yaml

ONLY_EXISTING_DIRS = True

def check_sim_dir_existance(config):
    # If config is a path to the config file, parse it first
    if isinstance(config, str):
      if not os.path.isfile(config):
        raise FileNotFoundError(f"Given config file {config} does not exist")
      config = parse_yaml(config)

    if "simulator_folder" in config:
      simulator_folder = os.path.normpath(config["simulator_folder"])
    else:
      simulator_folder = os.path.join(output_path(), config["simulator_name"])

    return os.path.isdir(simulator_folder)

def main():
    ## BUILD SIMULATOR (only required if source code is changed)

    # Define the path to a config file
    #config_file = "../configs/mobl_arms_index_pointing.yaml"
    _this_dir = os.path.dirname(os.path.realpath(__file__))
    _config_dir = os.path.join(_this_dir, "../configs/")
    print([os.path.join(_config_dir, f) for f in os.listdir(_config_dir) if f.endswith(".yaml")])
    for config_file in [os.path.join(_config_dir, f) for f in os.listdir(_config_dir) if f.endswith(".yaml")]:

        if ONLY_EXISTING_DIRS:
            sim_dir_exists = check_sim_dir_existance(config_file)
            print(f"{config_file.split('../')[-1]}: {'UPDATED' if sim_dir_exists else 'IGNORED'}")
            if not sim_dir_exists:
                continue
        
        # Build the simulator
        uitb.Simulator.build(config_file)

if __name__ == "__main__":
   main()