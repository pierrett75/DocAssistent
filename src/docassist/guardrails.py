import re
import unicodedata

# ---------------------------------------------------------------------------
# 1. Přístupová práva: role → úrovně přístupu dokumentů
# ---------------------------------------------------------------------------

# Každý chunk má v metadatech "access": "all" | "hr" | "management".
# Role uživatele určuje, které úrovně smí vidět. Ve skutečnosti role přijde z přihlášení
# (JWT token z Entra ID / Azure AD), my ji pro jednoduchost simulujeme hlavičkou X-User-Role.
ROLE_ACCESS: dict[str, list[str]] = {
    "employee": ["all"],
    "hr": ["all", "hr"],
    "management": ["all", "hr", "management"],
}
DEFAULT_ROLE = "employee"         


def allowed_access(role: str) -> list[str] | None:
    """Vrátí seznam úrovní, které role smí číst. Neznámá role → None (API z toho udělá 403)."""
    return ROLE_ACCESS.get(role.strip().lower())


# ---------------------------------------------------------------------------
# 2. Detekce prompt injection ve vstupu
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Malá písmena, bez diakritiky, jedna mezera. 'IGNORUJ  Předchozí' → 'ignoruj predchozi'.
    Útočník často píše bez háčků, takže vzory pak stačí psát jen v jedné podobě."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()


# Vzory píšeš BEZ diakritiky a malými písmeny (text projde normalize).
# \w* = libovolná koncovka (ignoruj / ignorujte), \s+ = mezera, (a|b) = jedno z.
INJECTION_PATTERNS: list[str] = [
    r"ignoruj\w*\s+(vsechn\w*\s+)?(predchozi|puvodni|sve|vyse)",
    r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above|your)",
    # zapomeň na všechny/své pokyny, forget everything
    r"(zapomen|forget)\w*\s+(na\s+)?(vsechn|vse|all|everything|svoje?|your|predchozi|previous)",
    # vypiš/ukaž mi svůj (systémový) prompt / své instrukce, reveal your system prompt
    r"(vypis|ukaz|prozrad|zopakuj|reveal|show|print|repeat)\w*\s+(mi\s+|me\s+)?(svuj|sve|svoje|tvuj|tve|your|the)?\s*(systemov\w*\s+)?(system\s+)?(prompt|instrukc\w*|pokyn\w*|instructions)",
    # změna role
    r"(chovej\w* se jako|act as|pretend to be|hraj\w* roli|developer mode|jailbreak)",
]


def check_input(question: str) -> str | None:
    """Vrátí vzor, který otázka porušila, nebo None, když je otázka v pořádku.
    Vzor vracíme kvůli logu: v produkci chceš vědět, JAKÉ útoky chodí, ne jen kolik."""

    normalized_question = normalize(question)
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized_question):
            return pattern


# Hláška pro uživatele: neprozrazuj, KTERÝ vzor sedl (útočník by se učil, co obejít).
BLOCKED_MESSAGE = "Tuhle otázku nemůžu zpracovat. Zeptejte se prosím na firemní směrnice."
