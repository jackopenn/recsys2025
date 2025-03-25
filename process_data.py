import os
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    # Data paths
    data_path: str = "data/ubc_data"
    output_path: str = "data"

    # Input files
    add_to_cart_path: Optional[str] = None
    page_visit_path: Optional[str] = None
    product_buy_path: Optional[str] = None
    remove_from_cart_path: Optional[str] = None
    search_query_path: Optional[str] = None
    product_properties_path: Optional[str] = None

    # Processing parameters
    user_events_chunk_size: int = 500000
    save_intermediate_files: bool = True

    def __post_init__(self):
        # Set default paths if not provided
        if not self.add_to_cart_path:
            self.add_to_cart_path = f"{self.data_path}/input/add_to_cart.parquet"
        if not self.page_visit_path:
            self.page_visit_path = f"{self.data_path}/input/page_visit.parquet"
        if not self.product_buy_path:
            self.product_buy_path = f"{self.data_path}/input/product_buy.parquet"
        if not self.remove_from_cart_path:
            self.remove_from_cart_path = (
                f"{self.data_path}/input/remove_from_cart.parquet"
            )
        if not self.search_query_path:
            self.search_query_path = f"{self.data_path}/input/search_query.parquet"
        if not self.product_properties_path:
            self.product_properties_path = (
                f"{self.data_path}/product_properties.parquet"
            )

        # Create output directories
        os.makedirs(f"{self.output_path}/user_events", exist_ok=True)
        os.makedirs(f"{self.output_path}/vocabs", exist_ok=True)


def load_data(config: Config):
    """Load input data files"""
    print("Loading data files...")

    # Load the data
    atc_df = pl.read_parquet(config.add_to_cart_path)
    pv_df = pl.read_parquet(config.page_visit_path)
    pb_df = pl.read_parquet(config.product_buy_path)
    rfc_df = pl.read_parquet(config.remove_from_cart_path)

    # Process search query data - extract numbers from query
    sq_df = pl.read_parquet(config.search_query_path).with_columns(
        pl.col("query").str.extract_all(r"\d+").cast(pl.List(pl.Int64))
    )

    # Process product properties - extract numbers from name
    pp_df = pl.read_parquet(config.product_properties_path).with_columns(
        pl.col("name").str.extract_all(r"\d+").cast(pl.List(pl.Int64))
    )

    return atc_df, pv_df, pb_df, rfc_df, sq_df, pp_df


def process_and_merge_data(atc_df, pv_df, pb_df, rfc_df, sq_df, pp_df, config: Config):
    """Process and merge all data sources"""
    print("Processing and merging data...")

    # Join each dataframe with product properties on the correct column
    atc_with_props = atc_df.join(pp_df, left_on="sku", right_on="sku", how="left")
    pb_with_props = pb_df.join(pp_df, left_on="sku", right_on="sku", how="left")
    rfc_with_props = rfc_df.join(pp_df, left_on="sku", right_on="sku", how="left")

    # Add event_type column to each dataframe before concatenation
    atc_with_props = atc_with_props.with_columns(
        pl.lit("add_to_cart").alias("event_type")
    )
    pb_with_props = pb_with_props.with_columns(
        pl.lit("product_buy").alias("event_type")
    )
    rfc_with_props = rfc_with_props.with_columns(
        pl.lit("remove_from_cart").alias("event_type")
    )
    sq_df = sq_df.with_columns(pl.lit("search_query").alias("event_type"))
    pv_df = pv_df.with_columns(pl.lit("page_visit").alias("event_type"))

    # Get all column names from the product events dataframes
    all_columns = set()
    for df in [atc_with_props, pb_with_props, rfc_with_props, pv_df, sq_df]:
        all_columns.update(df.columns)

    # Ensure all dataframes have the same schema by adding missing columns with null values
    for col in all_columns:
        # Add to cart
        if col not in atc_with_props.columns:
            atc_with_props = atc_with_props.with_columns(pl.lit(None).alias(col))
        # Product buy
        if col not in pb_with_props.columns:
            pb_with_props = pb_with_props.with_columns(pl.lit(None).alias(col))
        # Remove from cart
        if col not in rfc_with_props.columns:
            rfc_with_props = rfc_with_props.with_columns(pl.lit(None).alias(col))
        # Page visit
        if col not in pv_df.columns:
            pv_df = pv_df.with_columns(pl.lit(None).alias(col))
        # Search query
        if col not in sq_df.columns:
            sq_df = sq_df.with_columns(pl.lit(None).alias(col))

    # Create a list of dataframes for concatenation
    all_events = [atc_with_props, pb_with_props, rfc_with_props, pv_df, sq_df]
    all_events_df = pl.concat(all_events, how="diagonal_relaxed")

    # Fill null values and convert timestamps
    all_events_df = all_events_df.with_columns(
        [
            pl.col("timestamp").dt.epoch("s").alias("timestamp"),
            pl.col("url").fill_null(pl.lit(-1)).alias("url"),
            pl.col("name").fill_null(pl.lit([-1] * 16)).alias("name"),
            pl.col("query").fill_null(pl.lit([-1] * 16)).alias("query"),
            pl.col("category").fill_null(pl.lit(-1)).alias("category"),
            pl.col("price").fill_null(pl.lit(-1)).alias("price"),
            pl.col("sku").fill_null(pl.lit(-1)).alias("sku"),
        ]
    )

    # Save the combined events
    if config.save_intermediate_files:
        all_events_path = f"{config.output_path}/all_events.parquet"
        all_events_df.write_parquet(all_events_path)
        print(f"Saved all events to {all_events_path}")

    return all_events_df


def process_user_events(all_events_df, config: Config):
    """Process user events in chunks to avoid OOM issues"""
    print("Processing user events in chunks...")

    # Get unique client_ids
    unique_client_ids = all_events_df.select("client_id").unique().sort("client_id")
    total_clients = unique_client_ids.shape[0]
    print(f"Total unique clients to process: {total_clients}")

    # Define chunk size
    chunk_size = config.user_events_chunk_size
    num_chunks = (total_clients + chunk_size - 1) // chunk_size  # Ceiling division

    # Initialize an empty list to store paths to chunk files
    user_events_chunks = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_clients)

        print(
            f"Processing chunk {i+1}/{num_chunks} (clients {start_idx} to {end_idx-1})"
        )

        # Get client_ids for this chunk
        chunk_client_ids = unique_client_ids.slice(start_idx, end_idx - start_idx)

        # Filter events for these clients
        chunk_events = all_events_df.filter(
            pl.col("client_id").is_in(chunk_client_ids.select("client_id"))
        )

        # Group by client_id and aggregate
        chunk_user_events = (
            chunk_events.sort("timestamp")
            .group_by("client_id")
            .agg(
                pl.col("timestamp"),
                pl.col("event_type"),
                pl.col("sku"),
                pl.col("category"),
                pl.col("price"),
                pl.col("url"),  # page visit
                pl.col("name"),  # product
                pl.col("query"),  # search query
            )
        )

        # Save this chunk to disk
        chunk_file = f"{config.output_path}/user_events/chunk_{i}.parquet"
        chunk_user_events.write_parquet(chunk_file)

        # Add to our list
        user_events_chunks.append(chunk_file)

        print(f"Chunk {i+1} processed and saved to {chunk_file}")

    return user_events_chunks


def analyze_sequence_lengths(user_events_df):
    """Analyze the length of user event sequences"""
    print("Analyzing sequence lengths...")

    # Calculate the length of each user's event sequence
    user_sequence_lengths = (
        user_events_df.select(
            [
                pl.col("client_id"),
                pl.col("event_type").list.len().alias("sequence_length"),
            ]
        )
        .filter(pl.col("sequence_length") > 1)
        .filter(pl.col("sequence_length") < 105)
    )

    print(
        f"Count of users with sequence length > 1 and < 105: {user_sequence_lengths.shape[0]}"
    )

    # Group by sequence length and count users
    length_distribution = (
        user_sequence_lengths.group_by("sequence_length")
        .agg(pl.len().alias("user_count"))
        .sort("sequence_length")
    )

    print("Distribution of sequence lengths:")
    print(length_distribution.head(10))

    # Calculate some statistics
    stats = user_sequence_lengths.select(
        [
            pl.col("sequence_length").mean().alias("mean_length"),
            pl.col("sequence_length").median().alias("median_length"),
            pl.col("sequence_length").min().alias("min_length"),
            pl.col("sequence_length").max().alias("max_length"),
            pl.col("sequence_length").quantile(0.25).alias("q25_length"),
            pl.col("sequence_length").quantile(0.75).alias("q75_length"),
            pl.col("sequence_length").quantile(0.95).alias("q95_length"),
            pl.col("sequence_length").quantile(0.99).alias("q99_length"),
        ]
    )

    print("\nSequence length statistics:")
    print(stats)

    return length_distribution, stats


def build_vocabulary(df, column_name, min_freq=5, include_special_tokens=False):
    """
    Build vocabulary from a column containing lists of values.

    Args:
        df: Polars DataFrame
        column_name: Name of the column containing lists
        min_freq: Minimum frequency for an item to be included in vocabulary
        include_special_tokens: Whether to include special tokens like PAD, UNK

    Returns:
        Dictionary mapping tokens to indices and list of tokens
    """
    # Use polars to explode the lists and count frequencies
    value_counts = (
        df.select(pl.col(column_name).explode())
        .filter(pl.col(column_name).is_not_null())
        .group_by(column_name)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .filter(pl.col("count") >= min_freq)
    )

    # Get the filtered values as a list
    sorted_values = value_counts[column_name].to_list()

    # Create vocabulary with special tokens
    vocab = {}
    tokens = []

    idx = 0
    if include_special_tokens:
        # Add special tokens
        vocab["<PAD>"] = idx
        tokens.append("<PAD>")
        idx += 1

        vocab["<UNK>"] = idx
        tokens.append("<UNK>")
        idx += 1

    # Add regular tokens
    for value in sorted_values:
        vocab[value] = idx
        tokens.append(value)
        idx += 1

    return vocab, tokens


def build_vocabs(user_events_df, config: Config):
    """Build vocabularies for categorical features"""
    print("Building vocabularies for categorical features...")

    # Build vocabularies
    event_type_vocab, event_type_tokens = build_vocabulary(user_events_df, "event_type")
    sku_vocab, sku_tokens = build_vocabulary(user_events_df, "sku")
    category_vocab, category_tokens = build_vocabulary(user_events_df, "category")

    # Print vocabulary sizes
    print(f"Event type vocabulary size: {len(event_type_vocab)}")
    print(f"SKU vocabulary size: {len(sku_vocab)}")
    print(f"Category vocabulary size: {len(category_vocab)}")

    # Save vocabularies
    save_vocabulary(
        event_type_tokens, f"{config.output_path}/vocabs/event_type_vocab.txt"
    )
    save_vocabulary(sku_tokens, f"{config.output_path}/vocabs/sku_vocab.txt")
    save_vocabulary(category_tokens, f"{config.output_path}/vocabs/category_vocab.txt")

    return event_type_vocab, sku_vocab, category_vocab


def save_vocabulary(tokens, filepath):
    """Save vocabulary tokens to a text file"""
    with open(filepath, "w") as f:
        for token in tokens:
            f.write(f"{token}\n")
    print(f"Saved vocabulary to {filepath}")


def main(config: Config = None):
    """Main function to run the entire data processing pipeline"""
    if config is None:
        config = Config()

    print(f"Starting data processing with config: {config}")

    # Load data
    atc_df, pv_df, pb_df, rfc_df, sq_df, pp_df = load_data(config)

    # Process and merge data
    all_events_df = process_and_merge_data(
        atc_df, pv_df, pb_df, rfc_df, sq_df, pp_df, config
    )

    # Process user events in chunks
    user_events_chunks = process_user_events(all_events_df, config)

    # Read and concatenate user events for analysis
    # Note: In a real-world scenario with large data, we would analyze a sample
    user_events_path = f"{config.output_path}/user_events"
    user_events_df = pl.read_parquet(user_events_path)

    # Analyze sequence lengths
    length_distribution, stats = analyze_sequence_lengths(user_events_df)

    # Build vocabularies
    event_type_vocab, sku_vocab, category_vocab = build_vocabs(user_events_df, config)

    print("Data processing completed successfully!")


if __name__ == "__main__":
    main()
