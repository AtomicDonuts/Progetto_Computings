import numpy as np
from loguru import logger
import fits2csv as tested_module

DATASAMPLE = {
    "Source_Name": [np.str_("4FGL J0000.3-7355")],
    "DataRelease": [np.int16(1)],
    "RAJ2000": [np.float32(0.0983)],
    "DEJ2000": [np.float32(-73.922)],
    "GLON": [np.float32(307.70898)],
    "GLAT": [np.float32(-42.729538)],
    "Conf_68_SemiMajor": [np.float32(0.03237818)],
    "Conf_68_SemiMinor": [np.float32(0.03145309)],
    "Conf_68_PosAng": [np.float32(-62.7)],
    "Conf_95_SemiMajor": [np.float32(0.0525)],
    "Conf_95_SemiMinor": [np.float32(0.051)],
    "Conf_95_PosAng": [np.float32(-62.7)],
    "ROI_num": [np.int16(1726)],
    "Extended_Source_Name": [np.str_("")],
    "Signif_Avg": [np.float32(8.492646)],
    "Pivot_Energy": [np.float32(1917.7155)],
    "Flux1000": [np.float32(1.479606e-10)],
    "Unc_Flux1000": [np.float32(2.1770306e-11)],
    "Energy_Flux100": [np.float32(1.7352088e-12)],
    "Unc_Energy_Flux100": [np.float32(2.6915164e-13)],
    "SpectrumType": [np.str_("PowerLaw")],
    "PL_Flux_Density": [np.float32(4.2844265e-14)],
    "Unc_PL_Flux_Density": [np.float32(6.2686037e-15)],
    "PL_Index": [np.float32(2.2473958)],
    "Unc_PL_Index": [np.float32(0.117398046)],
    "LP_Flux_Density": [np.float32(4.8417162e-14)],
    "Unc_LP_Flux_Density": [np.float32(8.481274e-15)],
    "LP_Index": [np.float32(2.1280127)],
    "Unc_LP_Index": [np.float32(0.19689083)],
    "LP_beta": [np.float32(0.10999863)],
    "Unc_LP_beta": [np.float32(0.111856855)],
    "LP_SigCurv": [np.float32(1.1175609)],
    "LP_EPeak": [np.float32(1071.7054)],
    "Unc_LP_EPeak": [np.float32(1443.4996)],
    "PLEC_Flux_Density": [np.float32(4.8620054e-14)],
    "Unc_PLEC_Flux_Density": [np.float32(8.037235e-15)],
    "PLEC_IndexS": [np.float32(2.0488508)],
    "Unc_PLEC_IndexS": [np.float32(0.2069916)],
    "PLEC_ExpfactorS": [np.float32(0.18231755)],
    "Unc_PLEC_ExpfactorS": [np.float32(0.15101758)],
    "PLEC_Exp_Index": [np.float32(0.6666667)],
    "Unc_PLEC_Exp_Index": [np.float32(np.nan)],
    "PLEC_SigCurv": [np.float32(1.3292413)],
    "PLEC_EPeak": [np.float32(1427.5564)],
    "Unc_PLEC_EPeak": [np.float32(2230.5403)],
    "Npred": [np.float32(411.90985)],
    "Flux_Band": [
        np.array(
            [
                2.0565643e-08,
                1.2425028e-12,
                3.8831219e-10,
                1.3189076e-10,
                3.5006290e-11,
                5.3925649e-12,
                3.5607582e-16,
                5.1613483e-17,
            ],
            dtype=">f4",
        )
    ],
    "Unc_Flux_Band": [
        np.array(
            [
                [-1.6676287e-08, 1.4745653e-08],
                [np.nan, 1.6513468e-09],
                [-1.4248756e-10, 1.5454829e-10],
                [-2.6641196e-11, 2.8483122e-11],
                [-9.2036613e-12, 1.0352380e-11],
                [-2.6479136e-12, 3.7070728e-12],
                [np.nan, 2.2419759e-12],
                [np.nan, 2.0635616e-12],
            ],
            dtype=">f4",
        )
    ],
    "nuFnu_Band": [
        np.array(
            [
                3.2624888e-12,
                2.9143264e-16,
                2.5900609e-13,
                3.0935359e-13,
                2.3349362e-13,
                1.2648416e-13,
                2.3750427e-17,
                8.3339406e-18,
            ],
            dtype=">f4",
        )
    ],
    "Sqrt_TS_Band": [
        np.array(
            [1.1720686, 0.0, 2.7759964, 6.097597, 5.575633, 3.4596062, 0.0, 0.0],
            dtype=">f4",
        )
    ],
    "Variability_Index": [np.float32(12.834996)],
    "Frac_Variability": [np.float32(0.0)],
    "Unc_Frac_Variability": [np.float32(10.0)],
    "Signif_Peak": [np.float32(-np.inf)],
    "Flux_Peak": [np.float32(-np.inf)],
    "Unc_Flux_Peak": [np.float32(-np.inf)],
    "Time_Peak": [np.float64(-np.inf)],
    "Peak_Interval": [np.float32(-np.inf)],
    "Flux_History": [
        np.array(
            [
                6.0849459e-12,
                2.5016449e-09,
                2.8332263e-09,
                4.2172097e-09,
                2.4320108e-09,
                4.8514757e-09,
                4.2582977e-09,
                2.7345342e-09,
                1.7394171e-11,
                5.0010562e-10,
                2.0228490e-09,
                1.5150099e-09,
                2.4576792e-09,
                3.1096483e-09,
            ],
            dtype=">f4",
        )
    ],
    "Unc_Flux_History": [
        np.array(
            [
                [np.nan, 1.8864696e-09],
                [-1.5851342e-09, 1.7930387e-09],
                [-1.2066096e-09, 1.4523549e-09],
                [-1.3816318e-09, 1.5641212e-09],
                [-1.3739775e-09, 1.6136829e-09],
                [-1.6843601e-09, 1.9021948e-09],
                [-1.5829262e-09, 1.7982077e-09],
                [-1.1731729e-09, 1.4530912e-09],
                [np.nan, 1.6321493e-09],
                [np.nan, 1.5218908e-09],
                [-1.3911082e-09, 1.5667733e-09],
                [-9.2007785e-10, 1.1987534e-09],
                [-1.3403529e-09, 1.5400272e-09],
                [-1.4232091e-09, 1.5917939e-09],
            ],
            dtype=">f4",
        )
    ],
    "Sqrt_TS_History": [
        np.array(
            [
                0.0,
                1.6547282,
                3.6093166,
                3.99212,
                2.0243227,
                3.840539,
                3.172182,
                3.046058,
                0.0,
                0.41095558,
                1.5173048,
                1.9707487,
                2.0600674,
                2.4133372,
            ],
            dtype=">f4",
        )
    ],
    "ASSOC_4FGL": [np.str_("4FGL J0000.3-7355 ")],
    "ASSOC_FGL": [np.str_("")],
    "ASSOC_FHL": [np.str_("")],
    "ASSOC_GAM1": [np.str_("")],
    "ASSOC_GAM2": [np.str_("")],
    "ASSOC_GAM3": [np.str_("")],
    "TEVCAT_FLAG": [np.str_("N")],
    "ASSOC_TEV": [np.str_("")],
    "CLASS1": [np.str_("")],
    "CLASS2": [np.str_("")],
    "ASSOC1": [np.str_("")],
    "ASSOC2": [np.str_("")],
    "ASSOC_PROB_BAY": [np.float32(0.0)],
    "ASSOC_PROB_LR": [np.float32(0.0)],
    "RA_Counterpart": [np.float64(np.nan)],
    "DEC_Counterpart": [np.float64(np.nan)],
    "Unc_Counterpart": [np.float32(np.nan)],
    "Flags": [np.int16(0)],
}

def gen():
    logger.info("Generating FITS table..")
    tested_module.Table(DATASAMPLE).write("test_table.fits","fits")
    logger.info("Done.")

def test_import():
    try:
        pd = tested_module.fits_to_pandas(tested_module.custom_paths.Path("test_table.fits"))
    except:
        logger.error("Error!")
