import requests

def fetch_oeis_terms(sequence_terms=None, search_query=None, max_terms=1000):
    """
    Fetch terms of a sequence from OEIS using either sequence terms or a search query.
    Args:
        sequence_terms (list of int, optional): Terms of the sequence for searching.
        search_query (str, optional): Search term, keyword, or sequence description.
        max_terms (int): Maximum number of terms to retrieve, default is 1000.

    Returns:
        list: Sequence terms from OEIS, or None if not found.
    """
    base_url = "https://oeis.org/search"

    if sequence_terms:
        query = ",".join(map(str, sequence_terms))
        url = f"{base_url}?q={query}&fmt=json"
    elif search_query:
        url = f"{base_url}?q={search_query}&fmt=json"
    else:
        raise ValueError("Provide either sequence terms or a search query")

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        # Check if 'results' key is present in data
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
            if results and isinstance(results, list):
                # Get the first matching sequence
                sequence_info = results[0]
                terms = sequence_info.get("data", "")
                if terms:
                    # Convert terms from string to list of integers
                    terms_list = list(map(int, terms.split(",")))
                    return terms_list[:max_terms]  # Limit to max_terms if available
                else:
                    print("No terms found in sequence.")
                    return None
        else:
            # Print the raw JSON response for diagnosis
            print("Diagnostic response from OEIS API:", response.text)
            print("No sequence found.")
            return None
    else:
        print(f"Error fetching data from OEIS. Status code: {response.status_code}")
        return None

# Usage examples:
sequence_terms = [0, 1, 1, 2, 3, 5, 8, 13]  # Example for Fibonacci sequence
terms = fetch_oeis_terms(sequence_terms=sequence_terms)

if terms:
    print("Terms of the sequence (up to 1000):", terms)
else:
    print("Sequence not found or an error occurred.")
