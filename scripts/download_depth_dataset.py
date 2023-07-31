import os
import argparse

NOTICE = """
By accessing data via OpenTopography you agree to acknowledge OpenTopography and the dataset source as specified in the dataset metadata and on OpenTopography's data acknowledgement page in publications, presentations, and other materials produced using these data.

https://opentopography.org/citations
"""

S3_COMMAND = "aws s3 sync s3://raster/AW3D30/ {depth_path} --endpoint-url https://opentopography.s3.sdsc.edu --no-sign-request"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_path", type=str)
    args = parser.parse_args()
    print(NOTICE)
    cmd = S3_COMMAND.format(depth_path=args.depth_path)
    print("$", cmd)
    os.system(cmd)
