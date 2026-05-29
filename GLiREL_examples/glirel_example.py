import os
import sys
import spacy
import glirel

DEVICE = "cpu"
LABELS = {
    "glirel_labels": {

        # ── OCCUPATION / PROFESSION ──────────────────────────────────────────────
        "field of work": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP", "EVENT"]
        },
        "occupation": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "WORK_OF_ART"]
        },
        "profession": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG"]
        },
        "works as": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG"]
        },
        "specializes in": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP", "EVENT"]
        },
        "works in": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP", "EVENT", "LOC", "GPE"]
        },
        "expert in": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP", "EVENT"]
        },
        "specialist in": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP", "EVENT"]
        },
        # German equivalents
        "Beruf": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG"]
        },
        "arbeitete als": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG"]
        },
        "spezialisiert auf": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP", "EVENT"]
        },
        "Fachgebiet": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP", "EVENT"]
        },
        "Mikrobiologe": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP"]
        },
        "war ein": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP"]
        },

        # ── FOUNDER ──────────────────────────────────────────────────────────────
        "founder": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["ORG", "MISC", "NORP", "WORK_OF_ART", "EVENT"]
        },
        "founder of": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["ORG", "MISC", "NORP", "WORK_OF_ART", "EVENT"]
        },
        "founded by": {
            "allowed_tail": ["PERSON", "PER"],
            "allowed_head": ["ORG", "MISC", "NORP", "WORK_OF_ART", "EVENT"]
        },
        "co-founder": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["ORG", "MISC", "NORP", "WORK_OF_ART", "EVENT"]
        },
        "co-founder of": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["ORG", "MISC", "NORP", "WORK_OF_ART", "EVENT"]
        },
        "established": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["ORG", "MISC", "NORP", "WORK_OF_ART", "EVENT"]
        },
        "established by": {
            "allowed_tail": ["PERSON", "PER"],
            "allowed_head": ["ORG", "MISC", "NORP", "WORK_OF_ART", "EVENT"]
        },
        "regarded as founder of": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP", "WORK_OF_ART", "EVENT"]
        },
        "pioneer of": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP", "WORK_OF_ART", "EVENT"]
        },
        "father of": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP", "WORK_OF_ART", "EVENT"]
        },
        # German equivalents
        "Begründer": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP", "WORK_OF_ART", "EVENT"]
        },
        "Begründer von": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP", "WORK_OF_ART", "EVENT"]
        },
        "Begründer der": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP", "WORK_OF_ART", "EVENT"]
        },
        "gilt als Begründer": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP", "WORK_OF_ART", "EVENT"]
        },
        "Pionier": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP", "WORK_OF_ART", "EVENT"]
        },
        "Pionier der": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP", "WORK_OF_ART", "EVENT"]
        },
        "Vater der": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["MISC", "ORG", "NORP", "WORK_OF_ART", "EVENT"]
        },
        "gegründet von": {
            "allowed_tail": ["PERSON", "PER"],
            "allowed_head": ["ORG", "MISC", "NORP", "WORK_OF_ART", "EVENT"]
        },

        # ── DESIGN / INVENTION / CREATION ────────────────────────────────────────
        "designed by": {
            "allowed_head": ["FAC", "LOC", "WORK_OF_ART", "ORG", "PRODUCT", "MISC"],
            "allowed_tail": ["PERSON", "PER", "ORG"]
        },
        "designer of": {
            "allowed_head": ["PERSON", "PER", "ORG"],
            "allowed_tail": ["FAC", "LOC", "WORK_OF_ART", "PRODUCT", "MISC"]
        },
        "architect of": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["FAC", "LOC", "WORK_OF_ART", "ORG", "MISC"]
        },
        "architect": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["FAC", "LOC", "WORK_OF_ART", "ORG", "MISC"]
        },
        "invented by": {
            "allowed_head": ["PRODUCT", "MISC", "WORK_OF_ART", "ORG"],
            "allowed_tail": ["PERSON", "PER"]
        },
        "inventor of": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["PRODUCT", "MISC", "WORK_OF_ART", "ORG"]
        },
        "created by": {
            "allowed_tail": ["PERSON", "PER", "ORG"]
        },
        "creator of": {
            "allowed_head": ["PERSON", "PER", "ORG"]
        },
        "invention": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["PRODUCT", "MISC", "WORK_OF_ART", "ORG"]
        },
        "invented": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["PRODUCT", "MISC", "WORK_OF_ART", "ORG"]
        },
        "developed": {
            "allowed_head": ["PERSON", "PER", "ORG"],
            "allowed_tail": ["PRODUCT", "MISC", "WORK_OF_ART", "ORG"]
        },
        "developed by": {
            "allowed_tail": ["PERSON", "PER", "ORG"],
            "allowed_head": ["PRODUCT", "MISC", "WORK_OF_ART", "ORG"]
        },
        "created": {
            "allowed_head": ["PERSON", "PER", "ORG"]
        },
        "made": {
            "allowed_head": ["PERSON", "PER", "ORG"]
        },
        "built": {
            "allowed_head": ["PERSON", "PER", "ORG"],
            "allowed_tail": ["FAC", "LOC", "WORK_OF_ART", "PRODUCT", "MISC", "ORG"]
        },
        "built by": {
            "allowed_tail": ["PERSON", "PER", "ORG"],
            "allowed_head": ["FAC", "LOC", "WORK_OF_ART", "PRODUCT", "MISC", "ORG"]
        },
        # German equivalents
        "entworfen von": {
            "allowed_head": ["FAC", "LOC", "WORK_OF_ART", "ORG", "PRODUCT", "MISC"],
            "allowed_tail": ["PERSON", "PER", "ORG"]
        },
        "entworfen durch": {
            "allowed_head": ["FAC", "LOC", "WORK_OF_ART", "ORG", "PRODUCT", "MISC"],
            "allowed_tail": ["PERSON", "PER", "ORG"]
        },
        "Architekt": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["FAC", "LOC", "WORK_OF_ART", "ORG", "MISC"]
        },
        "Architekt von": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["FAC", "LOC", "WORK_OF_ART", "ORG", "MISC"]
        },
        "erbaut von": {
            "allowed_tail": ["PERSON", "PER", "ORG"],
            "allowed_head": ["FAC", "LOC", "WORK_OF_ART", "PRODUCT", "MISC", "ORG"]
        },
        "erfunden von": {
            "allowed_head": ["PRODUCT", "MISC", "WORK_OF_ART", "ORG"],
            "allowed_tail": ["PERSON", "PER"]
        },
        "erfand": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["PRODUCT", "MISC", "WORK_OF_ART", "ORG"]
        },
        "Erfinder": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["PRODUCT", "MISC", "WORK_OF_ART", "ORG"]
        },
        "Erfinder von": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["PRODUCT", "MISC", "WORK_OF_ART", "ORG"]
        },
        "entwickelt von": {
            "allowed_tail": ["PERSON", "PER", "ORG"],
            "allowed_head": ["PRODUCT", "MISC", "WORK_OF_ART", "ORG"]
        },
        "geschaffen von": {
            "allowed_tail": ["PERSON", "PER", "ORG"]
        },

        # ── CITIZENSHIP / NATIONALITY ─────────────────────────────────────────────
        "citizen of": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "NORP"]
        },
        "country of citizenship": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "NORP"]
        },
        "citizenship": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "NORP"]
        },
        "nationality": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "NORP"]
        },
        "national of": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "NORP"]
        },
        "became citizen of": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "NORP"]
        },
        "naturalized in": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "NORP"]
        },
        # German equivalents
        "Bürger von": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "NORP"]
        },
        "Staatsangehörigkeit": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "NORP"]
        },
        "wurde Bürger": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "NORP"]
        },
        "wurde Bürger von": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "NORP"]
        },
        "erhielt die Staatsbürgerschaft": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "NORP"]
        },
        "Nationalität": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "NORP"]
        },

        # ── PLACE OF BIRTH ────────────────────────────────────────────────────────
        "place of birth": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "FAC"]
        },
        "born in": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "FAC"]
        },
        "birthplace": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "FAC"]
        },
        # German equivalents
        "geboren in": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "FAC"]
        },
        "Geburtsort": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["LOC", "GPE", "FAC"]
        },

        # ── CAPITAL CITY ─────────────────────────────────────────────────────────
        "capital": {
            "allowed_head": ["GPE", "LOC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "capital of": {
            "allowed_head": ["GPE", "LOC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "capital city": {
            "allowed_head": ["GPE", "LOC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "capital city of": {
            "allowed_head": ["GPE", "LOC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "has capital": {
            "allowed_head": ["GPE", "LOC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "is capital of": {
            "allowed_head": ["GPE", "LOC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "seat of government of": {
            "allowed_head": ["GPE", "LOC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "largest city of": {
            "allowed_head": ["GPE", "LOC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        # German equivalents
        "Hauptstadt": {
            "allowed_head": ["GPE", "LOC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "Hauptstadt von": {
            "allowed_head": ["GPE", "LOC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "Hauptstadt und größte Stadt": {
            "allowed_head": ["GPE", "LOC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "ist die Hauptstadt": {
            "allowed_head": ["GPE", "LOC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "größte Stadt": {
            "allowed_head": ["GPE", "LOC"],
            "allowed_tail": ["GPE", "LOC"]
        },

        # ── LOCATION ─────────────────────────────────────────────────────────────
        "located in": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "WORK_OF_ART", "PERSON", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "location": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "WORK_OF_ART", "EVENT", "PERSON", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "location of": {
            "allowed_tail": ["GPE", "LOC", "FAC", "ORG", "WORK_OF_ART", "EVENT", "PERSON", "MISC"],
            "allowed_head": ["GPE", "LOC"]
        },
        "situated in": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "WORK_OF_ART", "PERSON", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "found in": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "WORK_OF_ART", "PERSON", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "in": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "WORK_OF_ART", "EVENT", "PERSON", "PER", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "based in": {
            "allowed_head": ["ORG", "PERSON", "PER"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "lies in": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "WORK_OF_ART", "PERSON", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "stands in": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "WORK_OF_ART", "PERSON", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        # German equivalents
        "befindet sich in": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "WORK_OF_ART", "PERSON", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "liegt in": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "WORK_OF_ART", "PERSON", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "steht in": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "WORK_OF_ART", "PERSON", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "befindet sich": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "WORK_OF_ART", "PERSON", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "sich befindet in": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "WORK_OF_ART", "PERSON", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "Standort": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "WORK_OF_ART", "EVENT", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },

        # ── RIVER / HYDROLOGY ─────────────────────────────────────────────────────
        "flows into": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "flows to": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "empties into": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "drains into": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "tributary of": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "mouth": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "mouth at": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "discharges into": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "runs into": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        # German equivalents
        "fließt in": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "mündet in": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "mündet in die": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "Mündung in": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "Nebenfluss von": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "fließt nach": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },

        # ── PART OF / MEMBERSHIP ─────────────────────────────────────────────────
        "part of": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "NORP", "MISC", "WORK_OF_ART"],
            "allowed_tail": ["GPE", "LOC", "FAC", "ORG", "NORP", "EVENT", "MISC"]
        },
        "belongs to": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "PRODUCT", "WORK_OF_ART", "MISC", "NORP"],
            "allowed_tail": ["GPE", "LOC", "ORG", "PERSON", "PER", "MISC", "NORP"]
        },
        "member of": {
            "allowed_head": ["PERSON", "PER", "GPE", "ORG", "MISC"],
            "allowed_tail": ["ORG", "NORP", "EVENT", "MISC"]
        },
        "component of": {
            "allowed_head": ["GPE", "LOC", "FAC", "PRODUCT", "MISC"],
            "allowed_tail": ["GPE", "LOC", "FAC", "ORG", "PRODUCT", "MISC"]
        },
        "contains": {
            "allowed_tail": ["GPE", "LOC", "FAC", "ORG", "PERSON", "PER", "MISC"],
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "MISC"]
        },
        "has part": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "MISC"],
            "allowed_tail": ["GPE", "LOC", "FAC", "ORG", "PERSON", "PER", "MISC"]
        },
        "within": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "MISC", "PERSON"],
            "allowed_tail": ["GPE", "LOC", "FAC", "ORG", "MISC"]
        },
        "inside": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "MISC", "PERSON"],
            "allowed_tail": ["GPE", "LOC", "FAC", "ORG", "MISC"]
        },
        # German equivalents
        "gehört zu": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "PRODUCT", "WORK_OF_ART", "MISC", "NORP"],
            "allowed_tail": ["GPE", "LOC", "FAC", "ORG", "NORP", "EVENT", "MISC"]
        },
        "gehört zum": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "PRODUCT", "WORK_OF_ART", "MISC", "NORP"],
            "allowed_tail": ["GPE", "LOC", "FAC", "ORG", "NORP", "EVENT", "MISC"]
        },
        "Teil von": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "NORP", "MISC", "WORK_OF_ART"],
            "allowed_tail": ["GPE", "LOC", "FAC", "ORG", "NORP", "EVENT", "MISC"]
        },
        "Teil der": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "NORP", "MISC", "WORK_OF_ART"],
            "allowed_tail": ["GPE", "LOC", "FAC", "ORG", "NORP", "EVENT", "MISC"]
        },
        "Mitglied von": {
            "allowed_head": ["PERSON", "PER", "GPE", "ORG", "MISC"],
            "allowed_tail": ["ORG", "NORP", "EVENT", "MISC"]
        },

        # ── ASSOCIATION / INVOLVEMENT ─────────────────────────────────────────────
        "associated with": {
            "allowed_head": ["PERSON", "PER", "ORG"],
            "allowed_tail": ["PERSON", "PER", "ORG", "EVENT", "NORP", "WORK_OF_ART", "MISC"]
        },
        "related to": {},
        "involved in": {
            "allowed_head": ["PERSON", "PER", "ORG", "GPE"],
            "allowed_tail": ["EVENT", "ORG", "NORP", "WORK_OF_ART", "MISC"]
        },
        "participated in": {
            "allowed_head": ["PERSON", "PER", "ORG", "GPE"],
            "allowed_tail": ["EVENT", "NORP", "MISC"]
        },
        "figure in": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "leader of": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["ORG", "GPE", "EVENT", "NORP", "MISC"]
        },
        "leading figure": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "leading figure in": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "leading figure of": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "prominent in": {
            "allowed_head": ["PERSON", "PER", "ORG"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "key figure in": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "key figure of": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "major figure in": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "major figure of": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "central figure in": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "influential in": {
            "allowed_head": ["PERSON", "PER", "ORG"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "representative of": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        # German equivalents
        "führende Figur": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "führende Figur des": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "führende Figur der": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "war eine führende Figur": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "Anführer": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["ORG", "GPE", "EVENT", "NORP", "MISC"]
        },
        "Vertreter": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "Vertreter des": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "Vertreter der": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "bedeutende Figur": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "Schlüsselfigur": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },
        "beteiligte sich an": {
            "allowed_head": ["PERSON", "PER", "ORG", "GPE"],
            "allowed_tail": ["EVENT", "NORP", "MISC"]
        },
        "nahm teil an": {
            "allowed_head": ["PERSON", "PER", "ORG", "GPE"],
            "allowed_tail": ["EVENT", "NORP", "MISC"]
        },
        "Parteienstreit": {
            "allowed_head": ["PERSON", "PER"],
            "allowed_tail": ["EVENT", "NORP", "ORG", "WORK_OF_ART", "MISC"]
        },

        # ── COUNTRY / REGION ─────────────────────────────────────────────────────
        "country": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "PERSON", "PER", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "country of": {
            "allowed_tail": ["GPE", "LOC", "FAC", "ORG", "MISC"],
            "allowed_head": ["GPE", "LOC"]
        },
        "in country": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "PERSON", "PER", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "state": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "region": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        # German equivalents
        "Land": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "PERSON", "PER", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "Bundesland": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "in Deutschland": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "WORK_OF_ART", "PERSON", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },
        "in Bayern": {
            "allowed_head": ["GPE", "LOC", "FAC", "ORG", "WORK_OF_ART", "PERSON", "MISC"],
            "allowed_tail": ["GPE", "LOC"]
        },

        # ── MOUNTAIN / MOUNTAIN RANGE ─────────────────────────────────────────────
        "mountain range": {
            "allowed_head": ["LOC", "GPE", "MISC", "ORG"],
            "allowed_tail": ["LOC", "GPE", "MISC", "NORP"]
        },
        "range": {
            "allowed_head": ["LOC", "GPE", "MISC", "ORG"],
            "allowed_tail": ["LOC", "GPE", "MISC", "NORP"]
        },
        "in range": {
            "allowed_head": ["LOC", "GPE", "MISC", "ORG"],
            "allowed_tail": ["LOC", "GPE", "MISC", "NORP"]
        },
        "in the Alps": {
            "allowed_head": ["LOC", "GPE", "FAC", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "highest peak": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "summit of": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        # German equivalents
        "Gebirge": {
            "allowed_head": ["LOC", "GPE", "MISC", "ORG"],
            "allowed_tail": ["LOC", "GPE", "MISC", "NORP"]
        },
        "Gebirgsgruppe": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "gehört zum Gebirge": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "in den Alpen": {
            "allowed_head": ["LOC", "GPE", "FAC", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "Alpen": {
            "allowed_head": ["LOC", "GPE", "FAC", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "Wettersteingebirge": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },
        "höchster Gipfel": {
            "allowed_head": ["LOC", "GPE", "MISC"],
            "allowed_tail": ["LOC", "GPE", "MISC"]
        },

        # ── EPISTEMIC / CLASSIFICATION ────────────────────────────────────────────
        "regarded as": {
            "allowed_head": ["PERSON", "PER", "ORG"],
            "allowed_tail": ["MISC", "NORP", "ORG", "EVENT"]
        },
        "considered": {
            "allowed_head": ["PERSON", "PER", "ORG"],
            "allowed_tail": ["MISC", "NORP", "ORG", "EVENT"]
        },
        "known as": {
            "allowed_head": ["PERSON", "PER", "ORG", "LOC", "GPE"],
            "allowed_tail": ["MISC", "NORP", "ORG", "EVENT"]
        },
        "recognized as": {
            "allowed_head": ["PERSON", "PER", "ORG"],
            "allowed_tail": ["MISC", "NORP", "ORG", "EVENT"]
        },
        "widely regarded as": {
            "allowed_head": ["PERSON", "PER", "ORG"],
            "allowed_tail": ["MISC", "NORP", "ORG", "EVENT"]
        },
        "seen as": {
            "allowed_head": ["PERSON", "PER", "ORG"],
            "allowed_tail": ["MISC", "NORP", "ORG", "EVENT"]
        },
        "described as": {
            "allowed_head": ["PERSON", "PER", "ORG"],
            "allowed_tail": ["MISC", "NORP", "ORG", "EVENT"]
        },
        # German equivalents
        "gilt als": {
            "allowed_head": ["PERSON", "PER", "ORG", "LOC", "GPE"],
            "allowed_tail": ["MISC", "NORP", "ORG", "EVENT", "WORK_OF_ART"]
        },
        "bekannt als": {
            "allowed_head": ["PERSON", "PER", "ORG", "LOC", "GPE"],
            "allowed_tail": ["MISC", "NORP", "ORG", "EVENT"]
        },
        "anerkannt als": {
            "allowed_head": ["PERSON", "PER", "ORG"],
            "allowed_tail": ["MISC", "NORP", "ORG", "EVENT"]
        },
        "betrachtet als": {
            "allowed_head": ["PERSON", "PER", "ORG"],
            "allowed_tail": ["MISC", "NORP", "ORG", "EVENT"]
        },
        "weithin als": {
            "allowed_head": ["PERSON", "PER", "ORG"],
            "allowed_tail": ["MISC", "NORP", "ORG", "EVENT"]
        },

        # ── CATCH-ALL ────────────────────────────────────────────────────────────
        "no relation": {}
    }
}

print("Loading English spaCy pipeline (en_core_web_lg)")
nlp_en = spacy.load("en_core_web_lg")
nlp_en.add_pipe("glirel", after="ner", config={"device": DEVICE})

print("Loading German spaCy pipeline (de_core_news_lg)")
nlp_de = spacy.load("de_core_news_lg")
nlp_de.add_pipe("glirel", after="ner", config={"device": DEVICE})

print("All models loaded successfully!\n")


def extract_and_print(nlp, text: str, lang_label: str, threshold: float = 0.0) -> None:
    docs = list(nlp.pipe([(text, LABELS)], as_tuples=True))
    doc = docs[0][0]

    # Show detected entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(f"  Entities : {entities if entities else '(none detected)'}")

    # Filter and sort relations
    relations = [r for r in doc._.relations if r["score"] >= threshold]
    relations_sorted = sorted(relations, key=lambda x: x["score"], reverse=True)

    if relations_sorted:
        print(f"  Relations (threshold >= {threshold}):")
        for r in relations_sorted[:20]:
            head = " ".join(r["head_text"])
            tail = " ".join(r["tail_text"])
            print(f"    {head!r:30s} --[{r['label']:35s}]--> {tail!r:25s} ({r['score']:.4f})")
    else:
        print(f"  Relations : (none above threshold {threshold})")


def run_bilingual_example(
    number: int,
    text_en: str,
    text_de: str,
    threshold: float = 0.05,
) -> None:
    print("=" * 80)
    print(f"EXAMPLE {number}")
    print("=" * 80)
    print(f"\n[ENGLISH] {text_en}")
    extract_and_print(nlp_en, text_en, "EN", threshold)
    print(f"\n[GERMAN]  {text_de}")
    extract_and_print(nlp_de, text_de, "DE", threshold)
    print()


# Example 1 – person / field of work
run_bilingual_example(
    1,
    "Robert Koch was a microbiologist who is widely regarded as a founder of modern bacteriology.",
    "Robert Koch war ein Mikrobiologe, der weithin als Begründer der modernen Bakteriologie gilt.",
)

# Example 2 – river / geography
run_bilingual_example(
    2,
    "The Elbe River flows into the North Sea.",
    "Die Elbe fließt in die Nordsee.",
)

# Example 3 – capital city
run_bilingual_example(
    3,
    "Berlin is the capital and largest city of Germany.",
    "Berlin ist die Hauptstadt und größte Stadt Deutschlands.",
)

# Example 4 – mountain / mountain range
run_bilingual_example(
    4,
    "The Zugspitze belongs to the Wetterstein mountain range in the Alps.",
    "Die Zugspitze gehört zum Wettersteingebirge in den Alpen.",
)

# Example 5 – building / location
run_bilingual_example(
    5,
    "The Cologne Cathedral is located in the city of Cologne, Germany.",
    "Der Kölner Dom befindet sich in der Stadt Köln in Deutschland.",
)

# Example 6 – designed by
run_bilingual_example(
    6,
    "The Brandenburg Gate was designed by architect Carl Gotthard Langhans.",
    "Das Brandenburger Tor wurde von dem Architekten Carl Gotthard Langhans entworfen.",
)

# Example 7 – invented by
run_bilingual_example(
    7,
    "Johannes Gutenberg invented the printing press in Europe.",
    "Johannes Gutenberg erfand die Druckpresse in Europa.",
)

# Example 8 – castle / location
run_bilingual_example(
    8,
    "Neuschwanstein Castle is located in Schwangau, Bavaria, Germany.",
    "Schloss Neuschwanstein befindet sich in Schwangau in Bayern, Deutschland.",
)

# Example 9 – citizenship
run_bilingual_example(
    9,
    "Albert Einstein became a citizen of Switzerland in 1901.",
    "Albert Einstein wurde 1901 Bürger der Schweiz.",
)

# Example 10 – person / movement
run_bilingual_example(
    10,
    "Richard Wagner was a leading figure of the War of the Romantics.",
    "Richard Wagner war eine führende Figur des Parteienstreits.",
)
