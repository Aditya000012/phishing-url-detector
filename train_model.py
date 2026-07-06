import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle
import os
from urllib.parse import urlparse
from feature_extractor import extract_features

def get_path_depth(u):
    try:
        p = urlparse(u if '://' in u else 'http://' + u)
        path = p.path.strip('/')
        return len([seg for seg in path.split('/') if seg]) if path else 0
    except Exception:
        # Fallback
        temp = u
        if '://' in temp:
            temp = temp.split('://', 1)[1]
        parts = temp.split('/', 1)
        if len(parts) > 1:
            path = parts[1].split('?')[0].strip('/')
            return len([seg for seg in path.split('/') if seg]) if path else 0
        return 0

def main():
    print("Fetching dataset...")
    url = "https://raw.githubusercontent.com/Jcharis/Detecting-Malicious-Url-With-Machine-Learning/master/urldata.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return
        
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    df = df.dropna(subset=['url', 'label'])
    
    # Extract path depth for the entire dataset to perform stratified sampling
    print("Computing path depths for all dataset records...")
    df = df.copy()
    df['path_depth'] = df['url'].apply(get_path_depth)
    df['path_depth_grouped'] = df['path_depth'].apply(lambda d: d if d < 5 else 5)
    
    # 1. DOUBLE-STRATIFICATION
    print("\nPerforming double-stratified sampling (3,000 samples per bucket)...")
    df_good_all = df[df['label'] == 'good']
    df_bad_all = df[df['label'] == 'bad']
    
    good_samples = []
    bad_samples = []
    
    for depth in range(6):
        # 0, 1, 2, 3, 4, 5+
        df_good_depth = df_good_all[df_good_all['path_depth_grouped'] == depth]
        df_bad_depth = df_bad_all[df_bad_all['path_depth_grouped'] == depth]
        
        good_samples.append(df_good_depth.sample(n=3000, random_state=42))
        bad_samples.append(df_bad_depth.sample(n=3000, random_state=42))
        
    df_good_stratified = pd.concat(good_samples)
    df_bad_stratified = pd.concat(bad_samples)
    
    df_final = pd.concat([df_good_stratified, df_bad_stratified]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Final training set size: {len(df_final)} (good: {len(df_good_stratified)}, bad: {len(df_bad_stratified)})")
    
    # 2. FEATURE EXTRACTION
    print("\nExtracting features from URLs (this may take a few seconds)...")
    features_list = []
    for idx, raw_url in enumerate(df_final['url']):
        features_list.append(extract_features(raw_url))
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1}/{len(df_final)} URLs...")
            
    feature_names = [
        'url_length', 'has_ip', 'count_dots', 'count_hyphens_in_domain', 'count_hyphens_in_path',
        'count_at_symbol', 'count_subdomains', 'has_https', 'has_double_slash_redirect', 'is_shortened',
        'has_suspicious_words_in_domain', 'has_suspicious_words_in_path', 'has_port_number', 'path_depth',
        'digit_ratio_in_domain'
    ]
    
    X = pd.DataFrame(features_list, columns=feature_names)
    y = (df_final['label'] == 'bad').astype(int) # 1 for phishing (bad), 0 for legitimate (good)
    
    # 3. PRINT DISTRIBUTION MATRICES
    print("\n=== 3a. FINAL PATH DEPTH DISTRIBUTION TABLE ===")
    df_dist = pd.DataFrame({'label': df_final['label'], 'path_depth': X['path_depth']})
    df_dist['path_depth_grouped'] = df_dist['path_depth'].apply(lambda d: str(d) if d < 5 else '5+')
    matrix_path = df_dist.groupby(['label', 'path_depth_grouped']).size().unstack(fill_value=0)
    print(matrix_path)
    
    # 4. TRAIN WITH REGULARIZATION
    print("\nSplitting dataset into 80/20 train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training regularized RandomForestClassifier (max_depth=10, min_samples_leaf=5)...")
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        min_samples_leaf=5, 
        random_state=42, 
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # 5. METRICS EVALUATION
    print("\nEvaluating model on test set...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("-" * 40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("-" * 40)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    
    # Feature importances
    importances = clf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importances:")
    print(importance_df.to_string(index=False))
    
    # Save feature importances to file
    with open('feature_importance.txt', 'w') as f:
        f.write("Feature Importances Ranked from Most to Least Important:\n")
        f.write("-" * 60 + "\n")
        f.write(importance_df.to_string(index=False))
        f.write("\n")
        
    # Save the trained model
    model_path = 'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"\nModel saved to '{model_path}' successfully.")
    
    # 6. RUN 30-URL BENIGN BATCH TEST
    print("\n=== 6. BATCH TEST ON 30 BENIGN URLS ===")
    benign_urls = [
        'https://accounts.google.com/signin',
        'https://github.com/settings/profile',
        'https://mail.google.com/mail/u/0/#inbox',
        'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'https://www.amazon.in/dp/B08L5VJYV5',
        'https://www.linkedin.com/feed',
        'https://www.linkedin.com/in/williamhgates',
        'https://docs.python.org/3/library/os.html',
        'https://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster-than-an-unsorted-array',
        'https://en.wikipedia.org/wiki/Artificial_intelligence',
        'https://www.nytimes.com/2026/07/04/world/europe/france-election.html',
        'https://github.com/tensorflow/tensorflow',
        'https://news.ycombinator.com/item?id=34567890',
        'https://www.reddit.com/r/science/comments/123456/new_discovery',
        'https://accounts.spotify.com/en/login',
        'https://www.microsoft.com/en-us/store/apps',
        'https://www.apple.com/shop/buy-iphone/iphone-15-pro',
        'https://www.nasa.gov/mission_pages/webb/main/index.html',
        'https://www.harvard.edu/about-harvard',
        'https://docs.microsoft.com/en-us/azure/devops',
        'https://www.netflix.com/youraccount',
        'https://slack.com/signin',
        'https://medium.com/@user/my-first-post-about-coding',
        'https://zoom.us/j/9876543210?pwd=abcde',
        'https://www.paypal.com/signin',
        'https://accounts.google.com/b/0/AddMailService',
        'https://github.com/dashboard',
        'https://assets.github.com/images/modules/dashboard/boot.js',
        'https://www.facebook.com/messages/t/12345678',
        'https://www.chase.com/personal/banking'
    ]
    features_benign = [extract_features(u) for u in benign_urls]
    df_benign = pd.DataFrame(features_benign, columns=feature_names)
    preds_benign = clf.predict(df_benign)
    probs_benign = clf.predict_proba(df_benign)
    
    fps = 0
    results_benign = []
    for url, feat, pred, prob in zip(benign_urls, features_benign, preds_benign, probs_benign):
        label = 'Phishing' if pred == 1 else 'Legitimate'
        conf = prob[pred] * 100
        if pred == 1:
            fps += 1
        results_benign.append({
            'URL': url[:50] + '...' if len(url) > 50 else url,
            'Prediction': label,
            'Confidence': f"{conf:.1f}%",
            'susp_domain': feat[10],
            'susp_path': feat[11],
            'hyphen_domain': feat[3],
            'hyphen_path': feat[4]
        })
    print(pd.DataFrame(results_benign).to_string(index=False))
    print(f"\nLegitimate FPR: {fps}/30 ({ (fps/30)*100 :.2f}%)")

    # 7. RUN HELD-OUT PHISHING BATCH TEST (31 URLs)
    print("\n=== 7. BATCH TEST ON 31 HELD-OUT PHISHING URLS ===")
    phish_urls = [
        'theivorycloset.com/magento/securedoc/verification.php?email=',
        'karinadoldan.com/wp-includes/SimplePie/Net/cgi-bin/',
        'sweetenglish.ir/stain/dropbox/domain/index.php',
        'meskatha.com/a0flf',
        'chainsforchange.com/Dropbox/file/files/db/file.php',
        'hip-pet.com/my.stuff',
        '6i3cb6owitcouepv.mywa2pay.com/-------',
        'casite-708442.cloudaccess.net/ap_suspected/amz/fce81aa4df3dfa105c93c4e0988cc26e/',
        'desertcast.com/news/feed/',
        'creditfinancialcompany.com/loan/verify',
        'bombasrivas.com.mx/actualizacion/dropbox/domain/',
        '176.103.55.73/chipdd2.exe',
        'sebastianalonso.com.ar/wp/new/',
        'brianzainformatica.it/15ox8nj',
        'coventgarden-florist.co.uk/vqmod/install/d-virus.php',
        'tigadiva.co.id/oo/hbb/hbb/hbb/index.php?userid=',
        'trudprom.ru/afdIJGY8766gyu',
        'masinaspalatpiese.com/G/G',
        'craftsreviews.com/',
        'cndoubleegret.com/admin/secure/cmd-login=727b21e847c151121d5a71df5f375f4d',
        'signorepizzaria.com.br/wp-content/plugins/6bacfed65c69e901e8112abf2f5d50b7/',
        'fra1.ib.adnxs.com/if?enc=mpmZmZmZuT9jEFg5tMi2PwAAAAAAAPA_YxBYObTItj-amZmZmZm5P7MrTkDC53sZVOzIWS2t4nA3SGpTAAAAALuiJwAhCQAAawYAAAIAAADJReYA8-4FAAAAAQBVU0QAVVNEANgCWgDPTQAAv40AAgUAAQIAAIwAsCYC-QAAAAA.&cnd=%21HCWVKQjrkOsBEMmLmQcYACDz3RcwADgAQABI6wxQu8WeAVgAYMQEaABwAHgAgAEAiAEAkAEBmAEBoAEBqAEDsAEAuQGamZmZmZm5P8EBmpmZmZmZuT_JAYpAX6veOQFA2QEAAAAAAADwP-ABAPUBAACAPg..&ccd=%21kwbePQjrkOsBEMmLmQcY890XIAA.&udj=uf%28%27a%27%2C+129071%2C+1399474231%29%3Buf%28%27c%27%2C+3852395%2C+1399474231%29%3Buf%28%27r%27%2C+15091145%2C+1399474231%29%3B&vpid=1153&apid=237123&referrer=http%3A%2F%2Ftreslagoas.olx.com.br&media_subtypes=1&ct=0&dlo=1',
        'urwij.pl/86e0b',
        'dropx.allon4dallas.com/a06e9e426fc521060ab17619adfd746/',
        'mccrarys.us/4GBrdf6',
        'www.insigmaus.com/wp-includes/pomo/dx.php?id=',
        'foresighttech.com/aspnet_client',
        'ultramarincentr.ru/ihreg',
        'sc.0openssl.com/',
        'outloook.eu.pn/outlook.html',
        'homestartstrexorcdn.wixstudio.com/bridge'
    ]
    features_phish = [extract_features(u) for u in phish_urls]
    df_phish = pd.DataFrame(features_phish, columns=feature_names)
    preds_phish = clf.predict(df_phish)
    probs_phish = clf.predict_proba(df_phish)
    
    fns = 0
    results_phish = []
    for url, feat, pred, prob in zip(phish_urls, features_phish, preds_phish, probs_phish):
        label = 'Phishing' if pred == 1 else 'Legitimate'
        conf = prob[pred] * 100
        if pred == 0:
            fns += 1
        results_phish.append({
            'URL': url[:50] + '...' if len(url) > 50 else url,
            'Prediction': label,
            'Confidence': f"{conf:.1f}%",
            'susp_domain': feat[10],
            'susp_path': feat[11],
            'hyphen_domain': feat[3],
            'hyphen_path': feat[4]
        })
    print(pd.DataFrame(results_phish).to_string(index=False))
    print(f"\nPhishing FNR: {fns}/31 ({ (fns/31)*100 :.2f}%)")

if __name__ == '__main__':
    main()
