'''

'''
import sys
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

sys.path.append("../imports/")
from custom_variables import *

def filter_dec(row):
    if np.isnan(row["spatial_DEC_value"]) and ~np.isnan(row["DEC"]):
        val = row["DEC"]
    else:
        val = row["spatial_DEC_value"]
    return val
def filter_ra(row):
    if np.isnan(row["spatial_RA_value"]) and ~np.isnan(row["RA"]):
        val = row["RA"]
    else:
        val = row["spatial_RA_value"]
    return val
database = pd.read_csv(csv_path)
coordinate = database[
    ["name", "DEC", "RA", "spatial_DEC_value", "spatial_RA_value"]
].copy()
coordinate["DEC"] = coordinate.apply(filter_dec, axis=1)
coordinate["RA"] = coordinate.apply(filter_ra, axis=1)
coordinate["RA_deg"] = coordinate.apply(lambda row: row["RA"] * u.deg, axis=1)  # type: ignore
coordinate["DEC_deg"] = coordinate.apply(lambda row: row["DEC"] * u.deg, axis = 1) # type: ignore
c = SkyCoord(ra=coordinate["RA_deg"], dec=coordinate["DEC_deg"], frame="icrs")
c_galactic = c.galactic
# coordinate["GLat"]  = c_galactic.l.wrap_at("180d").degree
coordinate["GLat"]  = c_galactic.l.degree
coordinate["GLong"] = c_galactic.b.degree
coordinate[["name","GLat","GLong"]].to_csv("../files/galattic_coordinates.csv",index= False)