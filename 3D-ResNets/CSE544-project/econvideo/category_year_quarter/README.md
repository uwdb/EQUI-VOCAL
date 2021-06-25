Pipeline breakdown:

Raw data:
Table 1: KiltsQuarterlyFinal.csv
Table 2: products.tsv

Script 1: gen_upc_price.py
	Retrieve two columns (upc, price) from Table 1, and get Table 1’

Script 2: generate_bin_range.py
	Join Table 1’ with Table 2, and get Table 3

Script 3: split_category.py
	Generate the bin range from Table 3
	Generate training data from Table 3, and get Table 3’
	Generate pairs of category indices and corresponding category names from Table 3

Script 4: train_lstm.py
	1st Run (Take argument 0):
Train the LSTM model and save corresponding checkpoints and plots
	2nd Run (Take argument 1):
Generate predicted price for training data (in-sample data), and get Table 4;
Generate predicted price for all data (out-sample data), and get Table 5;

Script 5: join.py
	Step 1: Join Table 4 with Table 5, and get Table 6 (without upc)
	Step 2: Join Table 6 with Table 3’ (to obtain upc), and get Table 7

Script 6: comp_rsquare.py
	Compute r^2 from Table 7 and generate the result for each category

#####################################################################################

KiltsQuarterlyFinal.csv
useful columns: [0, 2, 8, 12, -4, -2] ==> panel_year, quarter, upc_descr, product_group_descr, price, upc_actual_replace