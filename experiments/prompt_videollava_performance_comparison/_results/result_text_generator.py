# metrics_module.py

import pandas as pd

def write_metrics(metrics: dict):
    # Create a dictionary with your metrics
    # metrics = {
    #     'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    #     'Value': [0.95, 0.92, 0.93, 0.94]
    # }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(metrics)

    max_values = df.max()

    # Add an asterisk next to the max values
    for col in df.columns:
        df[col] = df[col].apply(lambda x: f'{x}*' if x == max_values[col] else x)

    # Convert the DataFrame to a string and split it into lines
    lines = df.to_string(index=False).split('\n')

    # Create a new list of lines with borders
    bordered_lines = ['| ' + line + ' |' for line in lines]

    # Add top and bottom borders
    top_border = '+' + '-' * (len(bordered_lines[0]) - 2) + '+'
    bordered_lines.insert(0, top_border)
    bordered_lines.append(top_border)

    # Write the bordered table to the txt file
    with open('metrics_table.txt', 'w') as f:
        f.write('\n'.join(bordered_lines))


