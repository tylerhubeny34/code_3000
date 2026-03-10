import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    data = {'anon_id': [], 'matched_name': []}
    for i in range(0, len(aux_df)):
        aux_name = aux_df.iloc[i, 0]
        pos_match = []
        for j in range(0, len(anon_df)):
            if aux_df.iloc[i, 1] == anon_df.iloc[j, 1]:
                pos_match.append(j)
        for k in range(2,4):
            for x in range(0, len(anon_df)):
                if (aux_df.iloc[i, k] != anon_df.iloc[x, k]) and (x in pos_match):
                    pos_match.remove(x)
        if len(pos_match) == 1:
            data['anon_id'].append(pos_match[0])
            data['matched_name'].append(aux_name)
    df = pd.DataFrame(data)
    return df


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return len(matches_df) / len(anon_df)
