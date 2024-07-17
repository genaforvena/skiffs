import os
import argparse
from util.text_utils import clean_text

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "src_file_path",
        type=str,
        help="Source file",
    )
    args.add_argument(
        "output_file_path",
        type=str,
        help="Output file",
    )
    args.add_argument(
        "--reorgonize",
        type=bool,
        default=False,
        help="Organize text using LDAClusterer",
    )

    src_file_path = args.parse_args().src_file_path
    src_file_path = os.path.abspath(src_file_path)
    output_file_path = args.parse_args().output_file_path
    output_file_path = os.path.abspath(output_file_path)

    clean_text(src_file_path, output_file_path)
    if args.parse_args().reorgonize:
        from util.text_utils import clean_and_organize_text

        clean_and_organize_text(output_file_path, output_file_path)
