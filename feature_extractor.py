from urllib.parse import urlparse
import ipaddress

def extract_features(url: str) -> list:
    """
    Extracts 15 lexical and structural features from a raw URL string,
    evaluating domain-level and path-level characteristics separately.
    Returns list of features in the exact order needed by the ML model.
    """
    if not url:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
    url_lower = url.lower()
    
    # 1. url_length
    url_length = len(url)
    
    # Preprocess URL for urllib parsing
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            parsed_url = urlparse('http://' + url)
        domain = parsed_url.netloc
        path = parsed_url.path
        query = parsed_url.query
    except ValueError:
        # Fallback if URL is malformed or has invalid bracketed netloc
        temp = url
        if '://' in temp:
            temp = temp.split('://', 1)[1]
        parts = temp.split('/', 1)
        domain = parts[0]
        path = '/' + parts[1] if len(parts) > 1 else ''
        query = ''
        
    # Strip port if present for domain-specific checks
    domain_no_port = domain
    if ':' in domain_no_port:
        domain_no_port = domain_no_port.split(':')[0]
        
    # Get remaining part of the URL (path, query, fragment)
    remaining_url = url.replace(parsed_url.scheme + '://', '', 1).replace(domain, '', 1)
        
    # 2. has_ip
    has_ip = 0
    try:
        ipaddress.ip_address(domain_no_port)
        has_ip = 1
    except ValueError:
        pass
        
    # 3. count_dots
    count_dots = url.count('.')
    
    # 4. count_hyphens_in_domain
    count_hyphens_in_domain = domain_no_port.count('-')
    
    # 5. count_hyphens_in_path
    count_hyphens_in_path = remaining_url.count('-')
    
    # 6. count_at_symbol
    count_at_symbol = 1 if '@' in url else 0
    
    # 7. count_subdomains
    if has_ip:
        count_subdomains = 0
    else:
        domain_parts = domain_no_port.split('.')
        count_subdomains = max(0, len(domain_parts) - 2)
        
    # 8. has_https
    # Evaluates to 1 ONLY when URL explicitly starts with https://, else 0
    has_https = 1 if url_lower.startswith('https://') else 0
    
    # 9. has_double_slash_redirect
    # Strip protocol prefix first (case-insensitively), then check for // in remaining
    temp_url = url
    temp_url_lower = temp_url.lower()
    if temp_url_lower.startswith('http://'):
        temp_url = temp_url[7:]
    elif temp_url_lower.startswith('https://'):
        temp_url = temp_url[8:]
    has_double_slash_redirect = 1 if '//' in temp_url else 0
    
    # 10. is_shortened
    shortening_services = {
        'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'rebrand.ly', 'is.gd', 'buff.ly',
        'adf.ly', 'bit.do', 'lnkd.in', 'db.tt', 'qr.ae', 'ow.ly', 'w.sharethis.com',
        'merky.de', 'hop.clickbank.net', 'shrinkee.com', 'tr.im', 'cli.gs', 'short.to',
        'budurl.com', 'moourl.com', 'snipurl.com', 'cur.lv', 'tiny.cc', 'short.ie'
    }
    domain_clean = domain_no_port.lower()
    if domain_clean.startswith('www.'):
        domain_clean = domain_clean[4:]
    is_shortened = 1 if domain_clean in shortening_services else 0
    
    # 11. has_suspicious_words_in_domain
    # 12. has_suspicious_words_in_path
    suspicious_words = {"secure", "account", "login", "verify", "update", "banking", "confirm"}
    has_suspicious_words_in_domain = 1 if any(word in domain_no_port.lower() for word in suspicious_words) else 0
    has_suspicious_words_in_path = 1 if any(word in remaining_url.lower() for word in suspicious_words) else 0
    
    # 13. has_port_number
    # Binary (0/1), whether domain portion contains a non-standard port (e.g. :8080)
    has_port_number = 0
    try:
        # Check if parsed_url has an explicit port and it is not standard 80/443
        if parsed_url.port is not None and parsed_url.port not in [80, 443]:
            has_port_number = 1
    except ValueError:
        # Fallback if port parsing failed but colon exists
        if ':' in domain:
            port_part = domain.split(':', 1)[1]
            if port_part.isdigit() and port_part not in ['80', '443']:
                has_port_number = 1
                
    # 14. path_depth
    # Count of /-separated path segments after domain (excluding query string)
    path_clean = path.strip('/')
    path_depth = len([p for p in path_clean.split('/') if p]) if path_clean else 0
    
    # 15. digit_ratio_in_domain
    # Proportion of numeric characters in domain name (excluding port)
    domain_len = len(domain_no_port)
    if domain_len > 0:
        digit_ratio_in_domain = round(sum(c.isdigit() for c in domain_no_port) / domain_len, 4)
    else:
        digit_ratio_in_domain = 0.0
        
    return [
        url_length,
        has_ip,
        count_dots,
        count_hyphens_in_domain,
        count_hyphens_in_path,
        count_at_symbol,
        count_subdomains,
        has_https,
        has_double_slash_redirect,
        is_shortened,
        has_suspicious_words_in_domain,
        has_suspicious_words_in_path,
        has_port_number,
        path_depth,
        digit_ratio_in_domain
    ]
