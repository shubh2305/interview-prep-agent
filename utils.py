def clean_response(response_raw):
    if response_raw.startswith('```json') and response_raw.endswith('```'):
        return response_raw.strip('`').strip('json\n')
    return response_raw