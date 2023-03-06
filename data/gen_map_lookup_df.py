# -*- coding: utf-8 -*-
"""
@author: tsdj

STATUS:
    Finalized for the time, but may be extended as additional columns/cells
    are requested.
"""


import pickle

from typing import List, Dict


def gen_map_name() -> Dict[str, str]:
    map_name = { # At 1, 2, 3, 4, 6, 9, 12 months
        # Home economic status (1)
        **{f'hj_oek{i + 1}': f'economyv{i}' for i in range(2, 9)},
        # Home harmony (2)
        **{f'hj_ha{i + 1}': f'harmonyv{i}' for i in (2, 3, 4, 5, 7, 8)},
        'hj_ha7': 'Harmonyv6',
        # Mother mental health (3)
        **{f'mor_psyk_kap{i + 1}': f'm_men_capv{i}' for i in range(2, 9)},
        # Mother physical health (4)
        'mor_fys_kap3': 'm_phys_capv2',
        'mor_fys_kap4': 'm_phys_capv3',
        'mor_fys_kap5': 'm_phys_capv4',
        'mor_fys_kap6': 'm_phys_capv5',
        'mor_fys_kap7': 'm_phys_capv6',
        'mor_fys_kap8': 'm_phys_capv7',
        'mor_fys_kap9': 'm_phys_capv8',
        # Mother hours worked in home (5)
        'm_arb_i_hj3': 'homeworkv1', # Starts early (1 instead of 2)
        'm_arb_i_hj4': 'homeworkv2',
        'm_arb_i_hj5': 'homeworkv3',
        'm_arb_i_hj6': 'homeworkv4',
        'm_arb_i_hj7': 'homeworkv6', # Then skips here
        'm_arb_i_hj8': 'homeworkv7',
        'm_arb_i_hj9': 'homeworkv8',
        # Mother hours worked outside home (6)
        'm_arb_uf_hj3': 'Work_outs_v1', # Start early at one, need skip later
        **{f'm_arb_uf_hj{i + 2}': f'work_outs_v{i}' for i in range(2, 5)},
        **{f'm_arb_uf_hj{i + 1}': f'work_outs_v{i}' for i in range(6, 9)},
        # Daycare (7)
        **{f'dagpl_vugst{i + 2}': f'daycarev{i}' for i in range(1, 5)}, # starts early
        **{f'dagpl_vugst{i + 1}': f'daycarev{i}' for i in range(6, 9)}, # then skup
        # Care and cleanliness (8)
        **{f'p_og_r{i + 2}': f'carev{i}' for i in range(1, 5)}, # starts early
        **{f'p_og_r{i + 1}': f'carev{i}' for i in range(6, 9)}, # then skips
        # Own bed (9)
        **{f'egen_seng{i + 1}': f'own_bed_v{i}' for i in range(2, 9)},
        # In air (10)
        **{f'i_luften{i + 1}': f'in_air_v{i}' for i in range(2, 9)},
        # Smiles (11)
        **{f'smiler{i + 1}': f'smiles_v{i}' for i in range(2, 9)},
        # Lifts head (12)
        **{f'baerer_hoved{i + 1}': f'lifts_head_v{i}' for i in range(2, 9)},
        # Babbles (13)
        **{f'pludrer{i + 1}': f'babbles_v{i}' for i in range(2, 9)},
        # Sits (14)
        **{f'sidder_alene{i + 1}': f'sits_v{i}' for i in range(2, 9)},
        # Breastfed (15)
        **{f'ernaering{i + 1}': f'bvf{i}' for i in range(2, 9)},
        # Number meals (16)
        **{f'ant_maaltid{i + 1}': f'nb_meals_v{i}' for i in range(2, 9)}
        }

    assert len(map_name) == 16 * 7

    assert len(set(map_name.keys())) == len(set(map_name.values()))

    # check all keys ends in 2-9
    check = {}

    for key in map_name.keys():
        prefix = key[:-1]
        number = int(key[-1])

        if prefix not in check:
            check[prefix] = {number}
        else:
            check[prefix] = check[prefix].union({number})

    for key, vals in check.items():
        assert vals == {3, 4, 5, 6, 7, 8, 9}, (key, vals)

    return map_name


def gen_col(name: str, numbers: List[int] = None):
    if numbers is None:
        numbers = list(range(3, 10))

    lookup_col_nb = {
        'hj_oek': 1,
        'hj_ha': 2,
        'mor_psyk_kap': 3,
        'mor_fys_kap': 4,
        'm_arb_i_hj': 5,
        'm_arb_uf_hj': 6,
        'dagpl_vugst': 7,
        'p_og_r': 8,
        'egen_seng': 9,
        'i_luften': 10,
        'smiler': 11,
        'baerer_hoved': 12,
        'pludrer': 13,
        'sidder_alene': 14,
        'ernaering': 15,
        'ant_maaltid': 16,
        }
    lookup_month = {
        3: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 6,
        8: 9,
        9: 12,
        }
    map_name = gen_map_name()
    base_str = 'tab-b-c{}-{}-mo'
    col_number = lookup_col_nb[name]

    result = {}

    for number in numbers:
        key = base_str.format(col_number, lookup_month[number])
        val = map_name[f'{name}{number}']

        result[key] = val

    return result


def main():
    """
    Creates and saves a mapping between the names used in used in the CIHVR
    data and the names used in this "project" for folder, files etc.

    This is used to map between the lookups used "internally" and the external
    data, such as in `gen_labels.py`.

    Parameters
    ----------
    None

    Returns
    -------
    None.

    """
    # Need to know the column name in the data frame - both in order to extract
    # the training data and in order to fill it back in.
    map_lookup_df = {
        # Yes, they do count from 2 - since 1 refers to a number not in Table B
        'weight-0-mo': 'bweight',
        'weight-1-mo': 'weightv2',
        'weight-2-mo': 'weightv3',
        'weight-3-mo': 'weightv4',
        'weight-4-mo': 'weightv5',
        'weight-6-mo': 'weightv6',
        'weight-9-mo': 'weightv7',
        'weight-12-mo': 'weightv8',
        # 'date-0-mo', # I dont think this exists, but in can be created from
        # the other variable we have, i.e. dob, mob, yob, even CPR.
        'date-1-mo': 'datev2',
        'date-2-mo': 'datev3',
        'date-3-mo': 'datev4',
        'date-4-mo': 'datev5',
        'date-6-mo': 'datev6',
        'date-9-mo': 'datev7',
        'date-12-mo': 'datev8',
        'length-0-mo': 'blength',
        'length-12-mo': 'length1y',
        # Table B small cells (1-16). Some of these names are from the column
        # names of the data frame, when not present in data frame tried to
        # create names of similar style.
        # Home economic status (1)
        **gen_col('hj_oek'),
        # Home harmony (2)
        **gen_col('hj_ha'),
        # Mother mental health (3)
        **gen_col('mor_psyk_kap'),
        # Mother physical health (4)
        **gen_col('mor_fys_kap'),
        # Mother hours worked in home (5)
        **gen_col('m_arb_i_hj'),
        # Mother hours worked outside home (6)
        **gen_col('m_arb_uf_hj'),
        # Daycare (7)
        **gen_col('dagpl_vugst'),
        # Care and cleanliness (8)
        **gen_col('p_og_r'),
        # Own bed (9)
        **gen_col('egen_seng'),
        # In air (10)
        **gen_col('i_luften'),
        # Smiles (11)
        **gen_col('smiler'),
        # Lifts head (12)
        **gen_col('baerer_hoved'),
        # Babbles (13)
        **gen_col('pludrer'),
        # Sits (14)
        **gen_col('sidder_alene'),
        # Breastfed (15)
        **gen_col('ernaering'),
        # Number meals (16)
        **gen_col('ant_maaltid'),
        # various
        'dura-any-breastfeed': 'bfdurany',
        'breastfeed-7-do': 'bfv1',
        'preterm-birth': 'preterm',
        'preterm-birth-weeks': 'pretermwk',
        # nurse names. note these are named by tsdj, see the following script
        # `~/data/prepare_nuse_name_data.py`
        'nurse-name-1': 'nurse-name-1',
        'nurse-name-2': 'nurse-name-2',
        'nurse-name-3': 'nurse-name-3',
        }

    assert len(map_lookup_df) == len(set(map_lookup_df.values()))

    with open('Y:/RegionH/Scripts/users/tsdj/storage/maps/map_lookup_df.pkl', 'wb') as file:
        pickle.dump(map_lookup_df, file)


if __name__ == '__main__':
    main()
