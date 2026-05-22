from urllib.parse import urlencode


def build_search_url(base_url, query, page):
    params = {
        "q": query.strip(),
        "page": str(page),
        "sort": "relevance",
    }
    return f"{base_url.rstrip('/')}?{urlencode(params)}"
