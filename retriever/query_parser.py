import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BoundAttribute:
    color: str
    garment: str
    confidence: float = 1.0


@dataclass
class ParsedQuery:
    original_query: str
    garments: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    bindings: List[BoundAttribute] = field(default_factory=list)
    environment: Optional[str] = None
    style: Optional[str] = None


class QueryParser:
    COLORS = {
        "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
        "white", "gray", "grey", "brown", "beige", "navy", "teal", "cream",
        "gold", "silver", "maroon", "burgundy", "olive",
    }

    GARMENTS = {
        "shirt": ["shirt", "tshirt", "t-shirt", "tee", "top", "blouse"],
        "tie": ["tie", "necktie"],
        "coat": ["coat", "jacket", "raincoat", "blazer", "cardigan"],
        "pants": ["pants", "trousers", "jeans", "leggings"],
        "dress": ["dress", "gown"],
        "skirt": ["skirt"],
        "shorts": ["shorts"],
        "shoes": ["shoes", "boots", "sneakers", "heels"],
        "bag": ["bag", "handbag", "purse", "backpack"],
        "hat": ["hat", "cap"],
        "scarf": ["scarf"],
    }

    ENVIRONMENTS = {
        "office": ["office", "business"],
        "street": ["street", "road", "sidewalk"],
        "park": ["park", "bench", "garden"],
        "home": ["home", "room", "house"],
        "indoor": ["indoor", "inside", "interior"],
        "outdoor": ["outdoor", "outside"],
        "city": ["city", "urban"],
        "formal": ["formal"],
    }

    STYLES = {
        "formal": ["formal", "professional", "business"],
        "casual": ["casual", "weekend", "relaxed"],
        "party": ["party", "wedding", "evening"],
        "sport": ["sport", "gym", "athletic"],
    }

    @classmethod
    def parse(cls, query: str) -> ParsedQuery:
        text = query.lower().replace("-", "")
        tokens = re.findall(r"\b[a-z]+\b", text)
        garment_positions = []
        color_positions = []

        for i, token in enumerate(tokens):
            color = "gray" if token == "grey" else token
            if color in cls.COLORS:
                color_positions.append((i, color))
            garment = cls._match_token(token, cls.GARMENTS)
            if garment:
                garment_positions.append((i, garment))

        garments = list(dict.fromkeys([g for _, g in garment_positions]))
        colors = list(dict.fromkeys([c for _, c in color_positions]))
        bindings = []

        for color_pos, color in color_positions:
            if not garment_positions:
                continue
            nearest_pos, nearest_garment = min(garment_positions, key=lambda x: abs(x[0] - color_pos))
            distance = abs(nearest_pos - color_pos)
            if distance <= 5:
                bindings.append(BoundAttribute(color=color, garment=nearest_garment, confidence=1.0 / (distance + 1)))

        environment = cls._match_any(tokens, cls.ENVIRONMENTS)
        style = cls._match_any(tokens, cls.STYLES)

        return ParsedQuery(
            original_query=query,
            garments=garments,
            colors=colors,
            bindings=bindings,
            environment=environment,
            style=style,
        )

    @classmethod
    def _match_token(cls, token: str, mapping) -> Optional[str]:
        for key, words in mapping.items():
            if token in words:
                return key
        return None

    @classmethod
    def _match_any(cls, tokens: List[str], mapping) -> Optional[str]:
        for token in tokens:
            match = cls._match_token(token, mapping)
            if match:
                return match
        return None
