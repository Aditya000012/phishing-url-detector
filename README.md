# Phishing URL Detector MVP - Final Walkthrough

This document outlines the final implementation, design decisions, debugging journey, performance metrics, and inherent structural limitations of the Phishing URL Detector MVP.

---

## 1. Problem Statement & MVP Scope

Phishing attacks commonly rely on deceptive URLs to trick users into revealing credentials or installing malware. The goal of this MVP is to build a lightweight, full-stack Flask web application that extracts lexical and structural features from a user-supplied URL and runs it through an offline machine learning model to predict whether it is **Phishing** or **Legitimate**, alongside a confidence percentage.

### MVP Scope Constraints:
*   **No Databases / Authentication**: Completely stateless.
*   **Offline Inference**: Feature extraction and model prediction run entirely locally via a pickled RandomForest classifier.
*   **Single-Page Utility**: Simple form input rendering detailed feature analysis and prediction results in real-time.
*   **Deployment-Ready**: Formatted for direct deployment to Render.

---

## 2. Final 15-Feature Set & Rationale

To accurately evaluate both domain-level spoofing (typosquatting) and path-level directory structure, we split the feature set into **15 lexical and structural features**:

1.  **`url_length`**: Total characters in the URL string.
2.  **`has_ip`**: Binary (0/1). Flagged if the hostname resolves to an IP address.
3.  **`count_dots`**: Number of dot (`.`) characters in the URL.
4.  **`count_hyphens_in_domain`**: Number of hyphens in the hostname (a strong typosquatting indicator).
5.  **`count_hyphens_in_path`**: Number of hyphens in the path/query/fragment (perfectly normal for deep content).
6.  **`count_at_symbol`**: Binary (0/1). Flagged if `@` exists (used to bypass browser host parsing).
7.  **`count_subdomains`**: Number of subdomains (excluding `www` and TLD).
8.  **`has_https`**: Binary (0/1). Evaluates to `1` only when the raw URL explicitly starts with `https://`.
9.  **`has_double_slash_redirect`**: Binary (0/1). Flagged if `//` appears anywhere in the URL after stripping the protocol prefix.
10. **`is_shortened`**: Binary (0/1). Flagged if the domain matches a list of known URL shortening services (e.g. `bit.ly`).
11. **`has_suspicious_words_in_domain`**: Binary (0/1). Substring match of keywords (`secure`, `account`, `login`, `verify`, `update`, `banking`, `confirm`) strictly within the hostname.
12. **`has_suspicious_words_in_path`**: Binary (0/1). Substring match of keywords in the path/query/fragment.
13. **`has_port_number`**: Binary (0/1). Flagged if a non-standard port (excluding 80/443) is specified.
14. **`path_depth`**: Count of `/`-separated segments in the URL path.
15. **`digit_ratio_in_domain`**: Proportion of numeric characters inside the domain portion (excluding port).

---

## 3. Model Architecture & Metrics

*   **Model Choice**: `RandomForestClassifier` (100 estimators).
*   **Regularization**: `max_depth=10`, `min_samples_leaf=5` (applied to prevent decision trees from fitting noisy split thresholds and step-functions on simple numeric bounds).
*   **Training Set**: **`36,000` real URLs** sourced from the Jcharis Malicious URL Dataset, balanced 50/50 and stratified on path depth.

### Final Model Evaluation Metrics (Test Set)
*   **Accuracy**: `73.88%`
*   **Precision**: `78.18%`
*   **Recall**: `65.47%`
*   **F1-Score**: `71.26%`

### Feature Importance Ranking
Separating domain and path features successfully isolated noise:
1.  `count_hyphens_in_path` — **30.04%**
2.  `has_suspicious_words_in_path` — **17.23%**
3.  `digit_ratio_in_domain` — **14.08%**
4.  `count_dots` — **11.82%**
5.  `url_length` — **10.60%**
6.  `path_depth` — **5.73%**
7.  `has_ip` — **2.58%**
8.  `count_hyphens_in_domain` — **2.55%**
9.  `count_subdomains` — **2.21%**
10. `count_at_symbol` — **1.18%**
11. `has_suspicious_words_in_domain` — **0.96%**
12. `has_port_number` — **0.49%**
13. `has_double_slash_redirect` — **0.36%**
14. `is_shortened` — **0.17%**
15. `has_https` — **0.00%**

---

## 4. Debugging Journey (Bugs Resolved)

During development, we uncovered and resolved three significant bugs:

### Bug 1: Protocol-Inference Mismatch
*   **Symptom**: URLs containing `https://` (like `https://www.youtube.com`) were predicted as Phishing.
*   **Root Cause**: The raw training dataset contained URLs with no protocol prefixes (e.g., `espdesign.com.au`), meaning `has_https` was a constant `0` during training. When real-world URLs starting with `https://` were tested at prediction time, `has_https` evaluated to `1`—a state the model associated with phishing since it never saw it in the legitimate training set.
*   **Fix**: Corrected the features logic to only evaluate `has_https` based on explicit prefixes and made `has_double_slash_redirect` strip protocol prefixes case-insensitively.

### Bug 2: Spurious Path-Depth Shortcut (Dataset Artifact)
*   **Symptom**: Legitimate deep-path URLs (like `github.com/dashboard`) were flagged as Phishing.
*   **Root Cause**: The dataset's Legitimate class was heavily skewed towards root domains (`path_depth = 0`), while Phishing URLs were concentrated at `path_depth = 1`. The model learned a dataset artifact shortcut: `depth 0` meant safe, `depth 1` meant phishing.
*   **Fix**: Implemented **double-stratified sampling**, pulling exactly 3,000 samples per path-depth bucket (0, 1, 2, 3, 4, 5+) for both classes. This brought the correlation of `path_depth` to the target to zero, forcing the model to ignore it as a shortcut and dropping its feature importance from `15.63%` to `5.99%`.

### Bug 3: Domain/Path Feature Conflation
*   **Symptom**: `accounts.google.com/b/0/AddMailService` was falsely predicted as Phishing at 92.6% confidence.
*   **Root Cause**: Suspicious word presence and hyphens were checked globally on the raw URL string. The subdomain `accounts` triggered `has_suspicious_words = 1`. Since legitimate sites in the dataset rarely triggered this feature globally, the presence of the word "accounts" inside `accounts.google.com` acted as an absolute proxy for phishing.
*   **Fix**: Separated `has_suspicious_words` and `count_hyphens` into domain-specific and path-specific features (resulting in 15 features). This dropped the importance of `has_suspicious_words_in_domain` to `0.96%`, allowing legitimate platforms with keywords in their hostnames to classify correctly.

### Bug 4 (Experiment): PhishTank Data Augmentation & Protocol Mismatch Reversion
*   **Symptom**: Attempted to add 6,000 verified phishing URLs from a PhishTank mirror dataset to train the model on disguised-phishing patterns (Wix-hosted templates or compromised WordPress folders). Upon retraining, the false positive rate on benign URLs surged to **`100.00%`**, and the false negative rate rose to **`64.52%`**.
*   **Root Cause**: URLs from PhishTank contain protocol prefixes (e.g. `https://`), while Jcharis URLs do not. Since Jcharis benign URLs evaluated to `has_https = 0` for 100% of samples, the model learned a massive spurious shortcut: `has_https = 1` always meant phishing. This caused the model to flag all legitimate `https://` URLs as phishing, and miss all phishing URLs presented without protocols.
*   **Fix & Engineering Decision**: We verified that even when balancing for this format difference, lexical features cannot distinguish disguised phishing because the hosting infrastructure's domain lexical structure (Wix, AWS, etc.) is clean. Adding more volume to a lexical model does not bypass this structural ceiling. We made the deliberate engineering decision to revert the model back to the stable 15-feature Jcharis-only baseline.

---

## 5. Known Limitations of Lexical-Only Classifiers

Lexical URL classification has an inherent performance ceiling. Two distinct failure modes were documented during batch testing:

### Limitation 1: False Positives on Legitimate Utility/Auth Pages (~33.3% Error Rate)
*   **Pattern**: Legitimate platforms using login/auth pages (e.g., `accounts.google.com/signin`, `accounts.spotify.com/en/login`, `netflix.com/youraccount`, `chase.com/personal/banking`) are flagged as Phishing.
*   **Cause**: The Jcharis dataset benign URLs were crawled from public web directories (like DMOZ), which catalog informational articles and root directories rather than user-interactive panels. As a result, only **0.18%** of Legitimate training URLs contain path keywords like `/login` or `/banking`, compared to **12.8%** of Phishing URLs.
*   **Takeaway**: In a lexical classifier, suspicious words in paths remain heavily weighted as phishing indicators, which inevitably false-positives legitimate utility pages.

### Limitation 2: False Negatives on Compromised Infrastructure (~26.7% Error Rate)
*   **Pattern**: Real phishing URLs hosted on legitimate platforms (e.g., WordPress directory compromises like `sweetenglish.ir/stain/dropbox/` or site-builder subdomains like `homestartstrexorcdn.wixstudio.com/bridge`) are misclassified as Legitimate.
*   **Cause**: The domain itself belongs to legitimate infrastructure, so it has no subdomains, hyphens, or digits. Furthermore, the folders used (e.g. `/bridge` or `/wp-content/`) avoid blacklist keywords.
*   **Takeaway**: The URL's lexical shape is identical to a standard legitimate web page. Lexical-only detection cannot detect phishing hosted on compromised infrastructure.

### How Real Systems Resolve This
Production anti-phishing software bypasses these limitations by adding:
1.  **Domain Whitelisting / Reputation Checks**: Instantly whitelisting high-authority domains (e.g., `google.com`) or checking domain registration age.
2.  **HTML/Visual Analysis**: Scraping the page to detect input forms, logo matches, or password entries.

---

## 6. Live Render URL

*   **Live Deployment URL**: `[Live Render URL will be inserted here upon git push and deployment]`
