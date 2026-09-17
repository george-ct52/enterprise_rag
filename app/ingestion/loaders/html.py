from bs4 import BeautifulSoup
import logfire 

def parse_html(file_path: str) -> str:
    """
    Parses HTML content and extracts text.

    Args:
        file_path (str): The path to the HTML file to parse.

    Returns:
        str: The extracted text from the HTML content.
    """
    with logfire.span("Parsing HTML file",filename=file_path):
        try:
            with open(file_path, 'r', encoding='utf-8',errors='ignore') as f:
                content = f.read()
            soup = BeautifulSoup(content, 'html.parser')

            # Remove script and style elements
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()

            # Extract text
            text = soup.get_text(separator='\n')

            # Collapse multiple newlines into a single newline
            text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())        

            logfire.info("HTML content parsed successfully.")
            return text
        
        except Exception as e:
            logfire.error(" X Error parsing HTML file.", exception=e)
            raise