import json
import os
import yaml
from datetime import datetime, timedelta
import google.generativeai as genai
from constants import SYSTEM_PROMPT
from colorama import Fore, Style
import re

def setup_gemini_api():
    """Setup Gemini API with your API key"""
    # Get API key from environment variable or config
    api_key = os.getenv('GEMINI_API_KEY')  # Set this in your environment
    if not api_key:
        print(f"{Fore.RED}❌ GEMINI_API_KEY not found in environment variables{Style.RESET_ALL}")
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    return model

def update_time_references_with_gemini(data_entry, gemini_model, current_date):
    """
    Use Gemini to intelligently update time references in training data
    """
    # Only process if there are existing startTime/endTime values
    filters = data_entry["data"]["filters"]
    if not (filters.get("startTime") or filters.get("endTime")):
        return data_entry
    
    query = data_entry["query"]
    current_date_str = current_date.strftime("%Y-%m-%d")
    
    # Create prompt for Gemini
    prompt = f"""
You are a date calculation assistant. Given a query and the current date, calculate the appropriate start and end times.

Current Date: {current_date_str}
Query: "{query}"

The query contains time references. Please calculate the exact start and end dates for this query.

IMPORTANT RULES:
- For "last/past X days/week/month" queries, calculate backwards from TODAY
- For "next X days/week/month" queries, calculate forwards from TODAY
- For calendar references like "last month" (without "past"), use actual calendar periods

Return ONLY a JSON object in this exact format:
{{
    "startTime": "YYYY-MM-DDTHH:MM:SS" or null,
    "endTime": "YYYY-MM-DDTHH:MM:SS" or null
}}

Examples:
- "emails from the last week" → start: 7 days ago 00:00:00, end: today 23:59:59
- "files from the past 3 days" → start: 3 days ago 00:00:00, end: today 23:59:59
- "meetings from last month" → start: first day of last calendar month 00:00:00, end: last day of last calendar month 23:59:59
- "documents from the last 30 days" → start: 30 days ago 00:00:00, end: today 23:59:59
- "today" → start: today 00:00:00, end: today 23:59:59
- "tomorrow" → start: tomorrow 00:00:00, end: tomorrow 23:59:59
- "next week" → start: tomorrow 00:00:00, end: 7 days from today 23:59:59

Be precise with calculations. Use 00:00:00 for start times and 23:59:59 for end times.
"""

    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Extract JSON from response
        # Find JSON object in the response
        json_match = re.search(r'\{[^}]*\}', response_text)
        if json_match:
            time_data = json.loads(json_match.group())
            
            # Update the data entry
            updated_entry = data_entry.copy()
            updated_entry["data"]["filters"]["startTime"] = time_data.get("startTime")
            updated_entry["data"]["filters"]["endTime"] = time_data.get("endTime")
            
            print(f"{Fore.GREEN}✅ Updated time references for: {query[:50]}...{Style.RESET_ALL}")
            return updated_entry
        else:
            print(f"{Fore.YELLOW}⚠️  No valid JSON found in Gemini response for: {query[:50]}...{Style.RESET_ALL}")
            return data_entry
            
    except Exception as e:
        print(f"{Fore.RED}❌ Error processing with Gemini: {e}{Style.RESET_ALL}")
        return data_entry

def format_data_for_finetuning(raw_json_data):
    """
    Converts a list of raw data entries into the format required for SFTTrainer.
    Each entry in the raw data should have a "query" and a "data" key.
    """
    current_date = datetime.now()
    
    # Setup Gemini API
    gemini_model = setup_gemini_api()
    if not gemini_model:
        print(f"{Fore.YELLOW}⚠️  Continuing without Gemini API - time references won't be updated{Style.RESET_ALL}")
    
    formatted_examples = []
    processed_count = 0
    
    for item in raw_json_data:
        # Update time references if Gemini is available and item has time fields
        if gemini_model:
            filters = item["data"]["filters"]
            if filters.get("startTime") or filters.get("endTime"):
                item = update_time_references_with_gemini(item, gemini_model, current_date)
                processed_count += 1
        
        user_query = item["query"]
        model_response_json = item["data"]

        # The model's expected output is the JSON object, converted to a string.
        output_json_string = json.dumps(model_response_json, separators=(',', ':'))

        # Construct the messages list for the chat template
        messages_list = [
            {"role": "user", "content": f"User Query: {user_query}\n\n{SYSTEM_PROMPT}"},
            {"role": "model", "content": output_json_string}
        ]
        formatted_examples.append({"messages": messages_list})

    if gemini_model and processed_count > 0:
        print(f"{Fore.CYAN}📅 Updated time references in {processed_count} queries using Gemini API{Style.RESET_ALL}")
    
    return formatted_examples

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), '../config/fine_tune_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        print(f"{Fore.CYAN}Loaded configuration from: {config_path}{Style.RESET_ALL}")

    # Define file paths from the config
    raw_data_path = os.path.join(os.path.dirname(__file__), '..', config["dataset_config"]["data_file"])
    processed_data_output_path = os.path.join(os.path.dirname(__file__), '..', config["dataset_config"]["processed_data_output"])

    print(f"Loading raw data from: {raw_data_path}")
    
    try:
        with open(raw_data_path, "r") as f:
            raw_json_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"{Fore.RED}❌ JSON Error: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🔧 Attempting to fix trailing comma...{Style.RESET_ALL}")
        
        # Read and fix the file
        with open(raw_data_path, "r") as f:
            content = f.read()
        
        # Remove trailing commas before closing brackets/braces
        import re
        fixed_content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        # Try parsing the fixed content
        raw_json_data = json.loads(fixed_content)
        print(f"{Fore.GREEN}✅ Successfully fixed and loaded data!{Style.RESET_ALL}")

    print(f"Processing {len(raw_json_data)} samples into chat format...")
    processed_data = format_data_for_finetuning(raw_json_data)

    print(f"Saving processed data to: {processed_data_output_path}")
    with open(processed_data_output_path, "w") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"{Fore.GREEN}Data preparation complete. Saved {len(processed_data)} samples.{Style.RESET_ALL}")
    
    # Display first 1-2 processed samples with colorful output
    samples_to_show = min(2, len(processed_data))
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Showing first {samples_to_show} processed sample(s):{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    for i in range(samples_to_show):
        entry = processed_data[i]
        
        print(f"\n{Fore.YELLOW}📋 SAMPLE {i+1}:{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'─'*40}{Style.RESET_ALL}")
        
        print(f"{Fore.MAGENTA}👤 USER CONTENT (input to model):{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{entry['messages'][0]['content'][:200]}...{Style.RESET_ALL}")
        
        print(f"\n{Fore.MAGENTA}🤖 MODEL CONTENT (expected output):{Style.RESET_ALL}")
        formatted_output = json.dumps(json.loads(entry['messages'][1]['content']), indent=2)
        print(f"{Fore.GREEN}{formatted_output}{Style.RESET_ALL}")
        
        print(f"{Fore.BLUE}{'─'*40}{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
