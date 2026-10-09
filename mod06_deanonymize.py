import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path, dtype={"zip3": str})
    aux = pd.read_csv(auxiliary_path, dtype={"zip3": str})
    return anon, aux

def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """

    identifiers = ["age", "zip3", "gender"]

    anon_counts = (anon_df.groupby(identifiers, dropna=False).size().reset_index(name = "anon_count"))
    aux_counts = (aux_df.groupby(identifiers, dropna=False).size().reset_index(name = "aux_count"))

    unique_anon = anon_df.merge(anon_counts[anon_counts["anon_count"] == 1][identifiers], on=identifiers, how="inner")
    unique_aux = anon_df.merge(aux_counts[aux_counts["aux_count"] == 1][identifiers], on=identifiers, how="inner")

    matches = unique_anon.merge(unique_aux, on=identifiers, how="inner", validate="one_to_one", suffixes=("_anon", "_aux"))

    return matches[["anon_id", "name"]].rename(columns={"name": "matched_name"})

def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0

    return matches_df["anon_id"].nunique() / len(anon_df)