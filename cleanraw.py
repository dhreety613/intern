def clean_people(contacts):

    original_count = len(
        contacts
    )

    with_email = []

    removed_no_email = 0

    for contact in contacts:

        email = contact.get(
            "email",
            ""
        ).strip()

        if not email:
            removed_no_email += 1
        else:
            with_email.append(
                contact
            )

    latest_contacts = {}

    for contact in with_email:

        email = contact[
            "email"
        ].strip().lower()

        latest_contacts[
            email
        ] = contact

    cleaned_contacts = list(
        latest_contacts.values()
    )

    removed_duplicates = (
        len(with_email)
        - len(cleaned_contacts)
    )

    return {
        "original_count": original_count,
        "removed_no_email": removed_no_email,
        "removed_duplicates": removed_duplicates,
        "final_count": len(cleaned_contacts),
        "people": cleaned_contacts
    }