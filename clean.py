import json

INPUT_FILE = "people_export.json"
OUTPUT_FILE = "people_cleaned.json"

print("\nLoading JSON...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    contacts = json.load(f)

original_count = len(contacts)

print(f"Original contacts loaded: {original_count:,}")

# -------------------------------
# STEP 1: REMOVE NO EMAIL
# -------------------------------

with_email = []
removed_no_email = 0

for contact in contacts:
    email = contact.get("email", "").strip()

    if not email:
        removed_no_email += 1
    else:
        with_email.append(contact)

print(f"Removed contacts without email: {removed_no_email:,}")
print(f"Remaining after email filter: {len(with_email):,}")

# -------------------------------
# STEP 2: KEEP LAST OCCURRENCE
# -------------------------------

latest_contacts = {}

for contact in with_email:
    email = contact["email"].strip().lower()

    # Later occurrences overwrite earlier ones
    latest_contacts[email] = contact

cleaned_contacts = list(latest_contacts.values())

removed_duplicates = len(with_email) - len(cleaned_contacts)

# -------------------------------
# SAVE CLEANED FILE
# -------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned_contacts, f, indent=2, ensure_ascii=False)

# -------------------------------
# SUMMARY
# -------------------------------

print("\n" + "=" * 50)
print("CLEANING COMPLETE")
print("=" * 50)

print(f"Original Contacts          : {original_count:,}")
print(f"Removed No Email           : {removed_no_email:,}")
print(f"Removed Duplicate Emails   : {removed_duplicates:,}")
print(f"Final Clean Contacts       : {len(cleaned_contacts):,}")

print(f"\nSaved File: {OUTPUT_FILE}")

# -------------------------------
# OPTIONAL SAMPLE OUTPUT
# -------------------------------

print("\nSample records:")

for contact in cleaned_contacts[:5]:
    print(
        f"{contact.get('name', '')} | "
        f"{contact.get('email', '')}"
    )