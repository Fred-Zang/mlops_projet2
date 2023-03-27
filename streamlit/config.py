"""

Config file for Streamlit App

"""

from member import Member


TITLE = "Le projet SatisPy"

TEAM_MEMBERS = [
    Member(
        name="Quan LIU",
        linkedin_url="https://www.linkedin.com/in/quan-liu-fr/",
        github_url="https://github.com/luckychien87",
    ),
    Member(
        name="Fred ZANGHI",
        linkedin_url="https://www.linkedin.com/in/fred-zanghi-89a01390/",
        github_url="https://github.com/Fred-Zang",
    ),

    Member(
        name="Eric GASNIER",
        linkedin_url="https://www.linkedin.com/in/ericgasnier/",
        github_url="https://github.com/egasnier",
    ),
]

PROMOTION = "Promotion ML Ops - Nov 2023"
