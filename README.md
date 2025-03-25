# Data Processing for RecSys Competition

This repository contains code for processing e-commerce event data for a recommender system.

## Overview

The main script `process_data.py` performs the following operations:

1. Loads data from parquet files (add to cart, page visit, product buy, remove from cart, search query, product properties)
2. Processes and merges all data sources into a unified event dataset
3. Groups events by user into sequences
4. Analyzes sequence lengths and distributions
5. Builds vocabularies for categorical features (event types, SKUs, categories)

## Usage

To run the data processing pipeline with default settings:

```bash
python process_data.py
```

### Configuration

You can customize the data processing by creating a `Config` object with your preferred settings:

```python
from process_data import Config, main

# Create custom config
config = Config(
    data_path="/path/to/data",
    output_path="/path/to/output",
    user_events_chunk_size=100000,
    save_intermediate_files=True
)

# Run with custom config
main(config)
```

### Configuration Options

- `data_path`: Base path to the input data files
- `output_path`: Base path for output files
- `add_to_cart_path`: Path to the add to cart events file (default: `{data_path}/input/add_to_cart.parquet`)
- `page_visit_path`: Path to the page visit events file (default: `{data_path}/input/page_visit.parquet`)
- `product_buy_path`: Path to the product buy events file (default: `{data_path}/input/product_buy.parquet`)
- `remove_from_cart_path`: Path to the remove from cart events file (default: `{data_path}/input/remove_from_cart.parquet`)
- `search_query_path`: Path to the search query events file (default: `{data_path}/input/search_query.parquet`)
- `product_properties_path`: Path to the product properties file (default: `{data_path}/product_properties.parquet`)
- `user_events_chunk_size`: Number of users to process in each chunk (default: 500000)
- `save_intermediate_files`: Whether to save intermediate files (default: True)

## Output Files

The script generates the following output files:

- `{output_path}/all_events.parquet`: Combined events from all data sources
- `{output_path}/user_events/chunk_{i}.parquet`: User event sequences, chunked by user ID
- `{output_path}/vocabs/event_type_vocab.txt`: Vocabulary for event types
- `{output_path}/vocabs/sku_vocab.txt`: Vocabulary for SKUs
- `{output_path}/vocabs/category_vocab.txt`: Vocabulary for categories

## Requirements

- Python 3.8+
- polars
- matplotlib
- numpy
