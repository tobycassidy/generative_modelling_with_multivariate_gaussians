import json
import pandas as pd

def build_taxonomy_tree(input_strings: list[str], include_other_tag: bool = False):
    tree = {}
    current_l1 = ""
    current_l2 = ""
    for input_string in input_strings:
        input_string_delimited = input_string.split("-")
        for i in range(len(input_string_delimited)):
            node_code = input_string_delimited[i]
            if i == 0:
                if not tree:
                    if include_other_tag:
                        tree = {
                            "name": node_code,
                            "children": [{"name": node_code + "-other"}],
                        }
                    else:
                        tree = {"name": node_code, "children": []}
            elif i == 1:
                if node_code != current_l1:
                    current_l1 = node_code
                    if include_other_tag:
                        tree["children"].append(
                            {
                                "name": current_l1,
                                "children": [{"name": current_l1 + "-other"}],
                            }
                        )
                    else:
                        tree["children"].append(
                            {"name": current_l1, "children": []}
                        )
            elif i == 2:
                if node_code != current_l2:
                    current_l2 = node_code
                    for j, child in enumerate(tree["children"]):
                        if child["name"] == current_l1:
                            if include_other_tag:
                                tree["children"][j]["children"].append(
                                    {
                                        "name": current_l2,
                                        "children": [
                                            {"name": current_l2 + "-other"}
                                        ],
                                    }
                                )
                            else:
                                tree["children"][j]["children"].append(
                                    {"name": current_l2, "children": []}
                                )
            elif i == 3:
                for j, child in enumerate(tree["children"]):
                    if child["name"] == current_l1:
                        for k, sub_child in enumerate(
                            tree["children"][j]["children"]
                        ):
                            if sub_child["name"] == current_l2:
                                tree["children"][j]["children"][k][
                                    "children"
                                ].append({"name": node_code})
    return tree

def process_node_codes(node_code_source):
    taxonomy_definition = []

    taxonomy_tree = {}
    df_node_code_source = pd.read_excel(node_code_source)
    df_node_code_source.columns = [
        col.lower().strip() if isinstance(col, str) else col
        for col in df_node_code_source.columns
    ]
    taxonomy_node_codes = sorted(
        list(set(df_node_code_source["taxonomy node code"].dropna()))
    )
    taxonomy_tree = build_taxonomy_tree(taxonomy_node_codes)
    taxonomy_definition.append(taxonomy_tree)
    return taxonomy_definition[0]

bob = process_node_codes('/Users/maximillianashton-lelliott/taxonomy-management/resources/taxonomies/aml/v1/aml_taxonomy_09092022.xlsx')
# Add to your existing Python script
js_file_path = '/Users/maximillianashton-lelliott/taxonomy-management/visualise/data.js'

data_dict = bob
json_data = json.dumps(data_dict)
js_line = f'var json_data = {json_data};\n'

with open(js_file_path, 'r') as file:
    lines = file.readlines()

for i, line in enumerate(lines):
    if line.startswith('var json_data ='):
        lines[i] = js_line  
        break
with open(js_file_path, 'w') as file:
    file.writelines(lines)
