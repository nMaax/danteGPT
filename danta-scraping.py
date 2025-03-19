import unicodedata
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import re  # Import regex for removing numbers

# Define a function to fix characters in the text
def fix_text(text):
    replacements = {
        'Ä': 'A', 'ä': 'a',
        'Ë': 'E', 'ë': 'e',
        'Ï': 'I', 'ï': 'i',
        'Ö': 'O', 'ö': 'o',
        'Ü': 'U', 'ü': 'u',
        '‘': "'", '’': "'",
        '“': '"', '”': '"',
        '«': '<<', '»': '>>'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Normalize text to remove any Unicode anomalies
    text = unicodedata.normalize("NFKC", text)

    # Remove BOM or other invisible characters
    text = text.replace("\ufeff", "").strip()

    return text

base_url = "https://digitaldante.columbia.edu"
sections = [
    {"name": "inferno", "cantos": 34},
    {"name": "purgatorio", "cantos": 33},
    {"name": "paradiso", "cantos": 33}
]

all_text = []
total_cantos = sum(section["cantos"] for section in sections)  # Total = 100

# Initialize progress bar with percentage
progress_bar = tqdm(total=total_cantos, desc="Scraping", unit="canto", 
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]")

for section in sections:
    section_name = section["name"]
    for canto in range(1, section["cantos"] + 1):
        url = f"{base_url}/dante/divine-comedy/{section_name}/{section_name}-{canto}/"
        
        try:
            # Fetch page with polite delay and headers
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Update progress bar description
            progress_bar.set_description(f"{section_name.capitalize()} {canto}".ljust(20))

            # Find Italian text container
            text_div = soup.find("div", class_="translation-entry")
            
            if text_div:
                italian_lines = []
                for p in text_div.find_all("p"):
                    # Extract full text from <p>, which includes both the number and the verse
                    full_text = p.get_text(" ", strip=True)  # Extract text with spaces

                    # Remove line numbers at the start of each line (they are usually 1-3 digits)
                    cleaned_text = re.sub(r'^\d+\s*', '', full_text)

                    # Fix characters
                    cleaned_text = fix_text(cleaned_text)

                    italian_lines.append(cleaned_text)

                canto_header = f"{section_name.upper()} CANTO {canto}"
                canto_header = fix_text(canto_header)
                canto_text = "\n".join(italian_lines)
                all_text.append(f"{canto_header}\n{canto_text}")
            else:
                all_text.append(f"!! ERROR: Text not found for {section_name} Canto {canto} !!")

            progress_bar.update(1)
            
        except Exception as e:
            all_text.append(f"!! FAILED: {section_name} Canto {canto} - {str(e)} !!")
        
        time.sleep(1)  # Respectful delay

progress_bar.close()

# Save to file with fixed text
with open("divina_commedia.txt", "w", encoding="utf-8") as f:
    fixed_text = fix_text("\n\n".join(all_text))
    f.write(fixed_text)

print("\nScraping complete! Check divina_commedia.txt")
