import os
import json
import glob
from collections import Counter

# The complete list of possible features derived from brute_force_selection.py
ALL_FEATURES = {
    'Closing_Momentum',
    'OBV_Slope',
    'Distance_to_Fast_SMA',
    'ATR_Percent',
    'Daily_RSI_14',
    'VWAP_Distance',
    'OFI',
    'Frac_Diff_Close',
    'Nifty_Momentum',
    'Nifty_RSI_14',
    'Nifty_Trend_Dist',
    'Gap_Percentage',
    'US_Overnight_Return'
}

def main():
    folder_path = "optimal_features"
    
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Error: The directory '{folder_path}' does not exist.")
        return

    used_features = set()
    feature_counts = Counter()
    
    json_files = glob.glob(os.path.join(folder_path, "*.json"))

    if not json_files:
        print(f"No JSON files found in the '{folder_path}' directory.")
        return

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            for combo in data.get("top_combinations", []):
                features = combo.get("features", [])
                for feature in features:
                    used_features.add(feature)
                    feature_counts[feature] += 1
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Determine unused features
    unused_features = ALL_FEATURES - used_features

    print(f"Analyzed {len(json_files)} JSON files.\n")
    
    if not unused_features:
        print(f"All the {len(ALL_FEATURES)} features are being used!")
    else:
        print(f"Unused features ({len(unused_features)}):")
        for i, feature in enumerate(sorted(unused_features), 1):
            print(f"{i}. {feature}")

    # Optional: Display top feature counts to see which ones are the most popular
    print("\n" + "-" * 40)
    print("Feature usage frequency:")
    print("-" * 40)
    for i, (feature, count) in enumerate(feature_counts.most_common(), 1):
        print(f"{i}. {feature}: {count} times")

if __name__ == "__main__":
    main()
