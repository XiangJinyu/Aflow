import pandas as pd
import glob


def compute_score_variance(csv_folder_path, n, output_csv='score_variance_sorted.csv'):
    """
    Compute the variance of the 'score' column for each index across multiple CSV files,
    and select the top N indices with the highest variance.

    Parameters:
    - csv_folder_path: Path to the folder containing CSV files.
    - top_n: The number of indices with the highest variance to be selected.
    - output_csv: Name of the output CSV file to save the sorted variance results.
    """
    # Get all CSV file paths in the specified folder
    csv_files = glob.glob(f"{csv_folder_path}/*.csv")

    if not csv_files:
        print("No CSV files found, please check the folder path and filenames.")
        return

    # Initialize an empty DataFrame to store all 'score' values
    score_df = pd.DataFrame()

    # Iterate through each CSV file and extract the 'score' column
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if 'score' not in df.columns:
                print(f"'score' column not found in file {file}, skipping this file.")
                continue
            score_df[file] = df['score']
        except Exception as e:
            print(f"Error occurred while reading file {file}: {e}")

    if score_df.empty:
        print("No valid 'score' data available for calculation.")
        return

    # Calculate the variance for each index across the 'score' columns
    score_variance = score_df.var(axis=1)

    # Add variance to the original index (assuming the index starts from 0)
    result = score_variance.reset_index()
    result.columns = ['index', 'variance']

    # Sort by variance in descending order
    result_sorted = result.sort_values(by='variance', ascending=False)

    # Select the top N indices with the highest variance
    top_variance = result_sorted.head(n)

    # Output the top N indices and their variance
    print(f"\nTop {n} indices with the highest variance:")
    print(top_variance)

    # Save the sorted variance results to a CSV file
    result_sorted.to_csv(output_csv, index=False)
    print(f"\nAll indices with variance have been saved to {output_csv}")

    # Extract the list of top N indices
    top_indices = top_variance['index'].tolist()
    print(f"\nList of top {n} indices with the highest variance:")
    print(top_indices)

    # Save the top N indices to a text file
    with open('top_indices.txt', 'w') as f:
        for idx in top_indices:
            f.write(f"{idx}\n")
    print("\nTop N indices have been saved to top_indices.txt")


if __name__ == "__main__":
    # Users can modify these parameters as needed
    csv_folder = 'ephemeral_data'  # Path to the folder containing CSV files
    top_n = 30  # Select the top N indices with the highest variance

    compute_score_variance(csv_folder_path=csv_folder, n=top_n)
