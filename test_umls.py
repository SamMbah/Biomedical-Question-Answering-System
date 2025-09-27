# test_umls.py

import os
from UmlsClient import UmlsClient

# 1) Instantiate and authenticate
client = UmlsClient(api_key=os.getenv("UMLS_API_KEY"))
tgt = client.authenticate()
print(" Obtained TGT:", tgt.split("/")[-1][:20] + "...")  # Print just the ticket suffix

# 2) Try a sample concept search
sample_text = "Type 1 diabetes hyperglycemia"
resp = client.find_cuis_for_text(sample_text)
print("\n Top search results for:", sample_text)
for hit in resp["result"]["results"][:3]:
    print(f"  CUI={hit['ui']}, Name={hit['name']}")

# 3) If we got at least one CUI, fetch its definitions properly
if resp["result"]["results"]:
    top_cui = resp["result"]["results"][0]["ui"]
    definitions_resp = client.get_cui_definitions(top_cui)

    defs_list = definitions_resp["result"]  # This should now be a list of dicts
    if isinstance(defs_list, list) and defs_list:
        print(f"\n Definitions for CUI {top_cui}:")
        # Print first 2 definitions (if available)
        for d in defs_list[:2]:
            val = d.get("value", "(no ‘value’ field)")
            src = d.get("rootSource", "(no source)")
            print(f"  [{src}] {val}")
    else:
        print(f"\n(No definitions found for CUI {top_cui})")
