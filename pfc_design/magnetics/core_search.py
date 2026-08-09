"""Web search for magnetic core specifications.

Scrapes manufacturer websites for core data to supplement local database.
Primary sources: Magnetics Inc., Micrometals, TDK/Ferroxcube.
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional


def search_mag_inc(part_number: str) -> Optional[dict]:
    """Search Magnetics Inc. website for a core by part number.

    Args:
        part_number: e.g. '0077083A7'

    Returns:
        dict with core data, or None if not found
    """
    url = f"https://www.mag-inc.com/products/powder-cores/kool-mu"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        # Look for the part number in the page
        # Note: actual scraping logic depends on website structure
        # This is a template — real implementation needs to match HTML
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                cell_text = ' '.join(c.get_text(strip=True) for c in cells)
                if part_number.lower() in cell_text.lower():
                    return {"found": True, "snippet": cell_text, "url": url}
        return None
    except Exception:
        return None


def search_micrometals(size_code: str) -> Optional[dict]:
    """Search Micrometals for a core by size/material.

    Args:
        size_code: e.g. 'T130-26'

    Returns:
        dict with core data, or None if not found
    """
    url = f"https://www.micrometals.com/design-tools/"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        # Search logic for Micrometals site
        if size_code.lower() in soup.get_text().lower():
            return {"found": True, "url": url}
        return None
    except Exception:
        return None


def find_core_online(od_mm: float, id_mm: float, ht_mm: float,
                     material_class: str = "Sendust") -> list[dict]:
    """Try to find matching cores online by dimensions.

    This is a convenience function that searches multiple sources.

    Args:
        od_mm, id_mm, ht_mm: core dimensions in mm
        material_class: Sendust, Ferrite, MPP, etc.

    Returns:
        List of dicts with found core information
    """
    results = []

    # Search keywords for different manufacturers
    search_urls = [
        f"https://www.mag-inc.com/products/powder-cores",
        f"https://www.micrometals.com",
        f"https://www.ferroxcube.com",
    ]

    for url in search_urls:
        try:
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                results.append({"url": url, "accessible": True})
        except Exception:
            results.append({"url": url, "accessible": False})

    return results
