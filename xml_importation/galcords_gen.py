''' 
Converte da Coordinate Equatoriali in Coordinate Galattiche il database
'''
import sys
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

# Import del modulo che contiene le variabili e i path della repository
sys.path.append("../imports/")
import custom_variables as custom_paths

database = pd.read_csv(custom_paths.csv_path)
coordinate = database[
    ["name", "DEC", "RA", "spatial_DEC_value", "spatial_RA_value"]
].copy()
coordinate["DEC"] = coordinate["spatial_DEC_value"].fillna(coordinate["DEC"])
coordinate["RA"] = coordinate["spatial_RA_value"].fillna(coordinate["RA"])


coordinate["RA_deg"] = coordinate.apply(lambda row: row["RA"] * u.deg, axis=1)  
coordinate["DEC_deg"] = coordinate.apply(lambda row: row["DEC"] * u.deg, axis = 1)


c = SkyCoord(ra=coordinate["RA_deg"], dec=coordinate["DEC_deg"], frame="icrs")
c_galactic = c.galactic

# coordinate["GLat"]  = c_galactic.l.wrap_at("180d").degree
coordinate["GLat"]  = c_galactic.l.degree
coordinate["GLong"] = c_galactic.b.degree


coordinate[["name","GLat","GLong"]].to_csv(custom_paths.gmap_path,index= False)
