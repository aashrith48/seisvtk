from seisvtk.io.segy_scan import detect_geometry, open_auto
import argparse
import os
from pathlib import Path

#Logging
import logging  
import sys
import segyio
import zarr
from seisvtk.io.logging_config import setup_logging
from seisvtk.io.segy2zarr import convert_segy_to_zarr

#Find relative path for output files
REPO_PATH = Path(__file__).resolve().parents[2]
output_folder=REPO_PATH / "Testing" / "Output"



def main():

    
    #Logger setup
    logger = logging.getLogger(__spec__.name if __spec__ else __name__)
    setup_logging()

    #Argument Parser (Setup sys.argv to receive script arguments)
    parser = argparse.ArgumentParser(description="Tool to automatically convert a SEGY to optimized format")
    parser.add_argument('filepath', help="Full file path to .segy file")
    parser.add_argument('-o','--out',help="Full output path for converted file")

    parser.add_argument('--auto', action='store_true', help='Autmatically scan SEGY for header information and geometry')

    parser.add_argument('--iline', type=int, help='Byte location for INLINE values in traces')
    parser.add_argument('--xline', type=int, help='Byte location for CROSSLINE values in traces')

    #Parse sys.argv
    args = parser.parse_args()

    #Normalizing output path 
    out = os.path.join(output_folder,args.out+'.zarr')

    #Automatically scan SEGY for geometry
    # Resolve inline/crossline bytes from either --auto or the manual flags.
    if args.auto:
        geom = detect_geometry(args.filepath)
        logger.info(f'Scanned SEGY Geometry: {geom}')
        iline, xline = geom["iline_byte"], geom["xline_byte"]
    elif args.iline is not None and args.xline is not None:
        iline, xline = args.iline, args.xline
        logger.info(f'Using INLINE:{iline} CROSSLINE:{xline} byte locations')
    else:
        parser.error('--iline and --xline are required when --auto is not set.')


    convert_segy_to_zarr(args.filepath, iline, xline, out, logger=logger)




if __name__ == '__main__':
    main()





