import os
import argparse
from tqdm import tqdm
import json
import requests
from retry import retry

def read_jsonL(file):
    data = []
    with open(file, "r", encoding='utf-8') as read_file:
        lines = read_file.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data

def write_jsonL(data, file):
    with open(file, "w", encoding='utf-8') as write_file:
        for line in data:
            json.dump(line, write_file, ensure_ascii=False)
            write_file.write("\n")


@retry(tries=3, delay=5, max_delay=60)
def get_wikipedia_candidates(mention, language='en', limit=10, fuzzy=False):
    """
    Given a mention string, return the top N candidate Wikipedia page IDs.

    Args:
        mention (str): The mention string to search.
        language (str): Language code for Wikipedia (default: 'en').
        limit (int): Number of candidates to return (default: 10).
        fuzzy (bool): Whether to enable fuzzy matching (default: False).

    Returns:
        List[Tuple[int, str]]: A list of (page_id, title) tuples.
    """
    endpoint = f"https://{language}.wikipedia.org/w/api.php"

    # Add fuzzy operator (~) if enabled
    search_query = f"{mention}~" if fuzzy else mention

    params = {
        "action": "query",
        "list": "search",
        "srsearch": search_query,
        "srlimit": limit,
        "format": "json"
    }

    headers = {
        # Wikipedia blocks requests without a descriptive User-Agent
        "User-Agent": "WikipediaCandidateFetcher/1.0 (contact: your_email@example.com)"
    }

    try:
        response = requests.get(endpoint, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        search_results = data.get("query", {}).get("search", [])

        candidates = [entry["pageid"] for entry in search_results]
        #candidates = [(entry["pageid"], entry["title"]) for entry in search_results]
        return candidates

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error fetching candidates for '{mention}': {e}")
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching candidates for '{mention}': {e}")
    except Exception as e:
        print(f"Unexpected error fetching candidates for '{mention}': {e}")

    return []

def generate_wikipedia_candidates(input_file, output_file, num_candidates=50, fuzzy=False):
    """
    Generate Wikipedia candidates for mentions in the input file.
    """
    input_data = read_jsonL(input_file)
    generated_candidates = []
    
    for i in tqdm(input_data, desc="Processing", unit="item"):
        #mention = i['mention']
        mention = i["LLM answer"].strip().replace("*", "") # adjust according to input file structure
        candidates = get_wikipedia_candidates(mention, limit=num_candidates, fuzzy=fuzzy) 
        #print(f"Mention: {mention}, Candidates: {candidates}")
        generated_candidates.append({
            "mention": mention,
            "candidates": candidates,
        })
    
    write_jsonL(generated_candidates, output_file)
    print(f"Results saved to {output_file}")



def main():
    parser = argparse.ArgumentParser(description='Query Wikidata to generate candidates.')
    parser.add_argument('--input_file', type=str, default= "LLM_output/gpt4_search_term/gpt_aida_new_serch_term.jsonl", help='Input file with mentions.')
    parser.add_argument('--output_folder', type=str, default= "regular_wp_search_for_new_search_term_gpt", help='Output folder for results.')
    parser.add_argument('--num_candidates', type=int, default=50, help='Number of candidates to return.')
    parser.add_argument('--fuzzy', action='store_true', help='Enable fuzzy search')
    parser.add_argument('--outfile_name', type=str, default="aida_gpt_search_term_cands", help="outfile name" )
    

    args = parser.parse_args()
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}")

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"Output folder {args.output_folder} created.")

    # Generate Wikipedia candidates
    generate_wikipedia_candidates(args.input_file, os.path.join(args.output_folder, f"wikipedia_candidates_{args.outfile_name}.jsonl"), num_candidates=args.num_candidates, fuzzy=args.fuzzy)
           
if __name__ == "__main__":
    main()