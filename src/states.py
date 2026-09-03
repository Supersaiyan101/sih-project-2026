"""Static geography for SIH26017 (Stage 0 design artifact).

Three states with real district names, 3-letter district codes, and approximate
centroid lat/lon. Used by data_generator.py to build the multi-state spatial model.

District codes are unique within a state. Village codes are assigned per district at
generation time; parcel IDs derive from (state_code, district_code, village_code,
khasra_no).
"""

STATES = {
    "Himachal Pradesh": {
        "code": "HP",
        "districts": {
            "Bilaspur": ("BIL", 31.33, 76.75),
            "Chamba": ("CHM", 32.55, 76.13),
            "Hamirpur": ("HAM", 31.68, 76.52),
            "Kangra": ("KNG", 32.10, 76.27),
            "Kinnaur": ("KIN", 31.60, 78.40),
            "Kullu": ("KUL", 31.96, 77.11),
            "Lahaul and Spiti": ("LAS", 32.50, 77.60),
            "Mandi": ("MAN", 31.71, 76.93),
            "Shimla": ("SHI", 31.10, 77.17),
            "Sirmaur": ("SIR", 30.75, 77.55),
            "Solan": ("SOL", 30.90, 77.10),
            "Una": ("UNA", 31.47, 76.27),
        },
    },
    "Punjab": {
        "code": "PB",
        "districts": {
            "Amritsar": ("AMR", 31.63, 74.87),
            "Barnala": ("BRL", 30.37, 75.55),
            "Bathinda": ("BTI", 30.21, 74.94),
            "Faridkot": ("FDK", 30.67, 74.76),
            "Fatehgarh Sahib": ("FGS", 30.64, 76.39),
            "Fazilka": ("FZK", 30.40, 74.03),
            "Ferozepur": ("FZR", 30.93, 74.61),
            "Gurdaspur": ("GSP", 32.04, 75.40),
            "Hoshiarpur": ("HSP", 31.53, 75.91),
            "Jalandhar": ("JAL", 31.33, 75.58),
            "Kapurthala": ("KPT", 31.38, 75.38),
            "Ludhiana": ("LDH", 30.90, 75.85),
            "Malerkotla": ("MLK", 30.53, 75.88),
            "Mansa": ("MNS", 29.99, 75.40),
            "Moga": ("MGA", 30.81, 75.17),
            "Sri Muktsar Sahib": ("MKT", 30.47, 74.52),
            "Pathankot": ("PTK", 32.27, 75.65),
            "Patiala": ("PTL", 30.34, 76.39),
            "Rupnagar": ("RUP", 30.97, 76.53),
            "SAS Nagar": ("SAS", 30.70, 76.72),
            "Sangrur": ("SGR", 30.24, 75.83),
            "SBS Nagar": ("SBS", 31.12, 76.13),
            "Tarn Taran": ("TTR", 31.45, 74.92),
        },
    },
    "Uttarakhand": {
        "code": "UK",
        "districts": {
            "Almora": ("ALM", 29.60, 79.66),
            "Bageshwar": ("BAG", 29.84, 79.77),
            "Chamoli": ("CHA", 30.41, 79.33),
            "Champawat": ("CMP", 29.33, 80.10),
            "Dehradun": ("DDN", 30.32, 78.03),
            "Haridwar": ("HRD", 29.95, 78.16),
            "Nainital": ("NAI", 29.38, 79.46),
            "Pauri Garhwal": ("PGR", 30.15, 78.78),
            "Pithoragarh": ("PTH", 29.58, 80.21),
            "Rudraprayag": ("RDP", 30.29, 78.98),
            "Tehri Garhwal": ("TGR", 30.38, 78.48),
            "Udham Singh Nagar": ("USN", 28.98, 79.41),
            "Uttarkashi": ("UTK", 30.73, 78.44),
        },
    },
}

# project type -> short code (used in project ID <STATE>-<TYPE>-<SEQ>)
PROJECT_TYPE_CODES = {
    "road": "RDH",
    "rail": "RLY",
    "irrigation": "IRR",
    "dam": "DAM",
    "industrial": "IND",
}
