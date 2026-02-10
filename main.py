import argparse
import yaml
from matching_pipeline import run_pipeline  

def match_clouds(cloud1, cloud2, config):
    """
    Public API.
    """
    if isinstance(config, str):
        cfg = yaml.safe_load(open(config))
    else:
        cfg = config

    return run_pipeline(cloud1, cloud2, cfg)


def main():
    """
    CLI wrapper.
    """

    parser = argparse.ArgumentParser(description="LiMatch CLI unified")
    parser.add_argument("-y", "--yml", type=str, required=True,
                        help="Path to YAML configuration file")
    parser.add_argument("-c1", "--cloud1", type=str, required=True,
                        help="Cloud A")
    parser.add_argument("-c2", "--cloud2", type=str, required=True,
                        help="Cloud B")

    args = parser.parse_args()

    corres, stats = match_clouds(args.cloud1, args.cloud2, args.yml)
    
if __name__ == "__main__":
    main()
