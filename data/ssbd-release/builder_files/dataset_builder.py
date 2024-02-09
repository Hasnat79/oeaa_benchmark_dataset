import tabula
import json

def write_json(filename, data):
    return json.dump(data, open(filename,'w'),indent=4)

data_list = "/scratch/user/hasnat.md.abdullah/oeaa_benchmark_dataset/data/ssbd-release/url-list.pdf"

df = tabula.read_pdf(data_list,pages='all')


res = {}

#extraction from xml
# continue from here **


# import xml.etree.ElementTree as ET

# # Parse the XML file
# tree = ET.parse('path_to_your_file.xml')

# # Get the root element
# root = tree.getroot()

# # Traverse the XML tree
# for child in root:
#     print(f"{child.tag}: {child.text}")
#     for subchild in child:
#         print(f"  {subchild.tag}: {subchild.text}")
write_json('ssbbd_benchmark.json', res)