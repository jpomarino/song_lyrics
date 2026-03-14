import pandas as pd


def filter_songs_by_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes songs that have certain keywords in the title.

    Args:
        df (pd.DataFrame): DataFrame to filter

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    df = df[
        ~df["title"].str.contains(
            r"remix|acoustic|version|cover|commentary|disney|spotify|demo|clean edit|radio edit|a cappella|sped up|extended dance break|slowed down|(live |parody|mixed|medley|session)|voicenote|recorded at|acapella|apple music|amazon|ISOLATED VOCALS|SLOWED & REVERB|tour|voice note|live\)| mix|\*",
            case=False,
        )
    ]
    return df


def filter_songs_manually(df: pd.DataFrame) -> pd.DataFrame:

    # 1. Charli xcx (didn't need to filter)

    # 2. Sabrina Carpenter
    df = df[
        ~((df["artist"] == "Sabrina Carpenter") & (df["title"] == "Thinking Out Loud"))
    ]

    # 3. Audrey Hobert (no need to filter)

    # 4. Reneé Rapp (no need to filter)

    # 5. Holly Humberstone (no need to filter)

    # 6. Maisie Peters (no need to filter)

    # 7. Carly Rae Jepsen (no need to filter)

    # 8. Lorde
    df = df[df["album"] != "Te Ao Mārama"]

    # 9. Addison Rae
    df = df[~df["album"].str.contains(r"diet pepsi|aquamarine", case=False)]

    # 10. Billie Eilish
    df = df[df["album"] != "Billie Eilish"]
    df = df[
        ~df["title"].str.contains(
            r"Billie Eilish|Lo Vas A Olvidar|hotline bling|edit", case=False
        )
    ]

    # 11. Olivia Rodrigo
    df = df[
        ~df["album"].str.contains(
            r"Bizaardvark|high school|stick season|disney", case=False
        )
    ]

    # 12. Maggie Rogers (no need to filter)

    # 13. Gracie Abrams
    df = df[~df["title"].str.contains("Gracie Abrams", case=False)]

    # 14. Tate McRae
    df = df[~df["title"].str.contains(r"sped-up|slowed", case=False)]

    # 15. Chappell Roan (no need to filter)

    # 16. Olivia Dean (no need to filter)

    # 17. Rachel Chinouriri (no need to filter)

    # 18. Ethel Cain (no need to filter)

    # 19. Kacey Musgraves
    df = df[~df["title"].str.contains(r"gracias|feliz", case=False)]

    # 20. Ariana Grande
    df = df[df["album"] != "Ariana Grande"]
    df = df[
        ~(
            (df["artist"] == "Ariana Grande")
            & (
                df["title"].str.contains(
                    r"edit|a capella|slowed|bonus|radio|reprise|japanese", case=False
                )
            )
        )
    ]

    return df


def filter_short_songs(df: pd.DataFrame, min_words: int = 30) -> pd.DataFrame:
    df = df[df["lyrics"].str.split().apply(len) >= min_words]
    return df


df = pd.read_json("../data/raw/lyrics_raw.json")
df = filter_songs_by_title(df)
df = filter_songs_manually(df)
df = df[~df.duplicated(subset=["artist", "title"])]
df = filter_short_songs(df)
df = df.reset_index(drop=True)

print(f"Saving a total of {len(df)} songs.")

# Save records as a JSON file
df.to_json("../data/processed/filtered_songs.json", indent=4, orient="records")
