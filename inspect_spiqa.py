#The output will tell us the actual structure of your downloaded SPIQA file.

# import json

# file_path = "data/test-A/SPIQA_testA.json"

# with open(file_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# print("Type of data:", type(data))

# if isinstance(data, dict):
#     print("Number of entries:", len(data))
#     print("\nTop-level keys:")
#     print(data.keys())

# elif isinstance(data, list):
#     print("Number of entries:", len(data))
#     print("\nFirst entry:")
#     print(data[0])


# the output will tell us the ThePaper ID, Type of paper data, Keys inside this paper


# import json

# file_path = "data/test-A/SPIQA_testA.json"

# with open(file_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Get the first paper
# paper_id = list(data.keys())[0]
# paper = data[paper_id]

# print("Paper ID:")
# print(paper_id)

# print("\nType of paper data:")
# print(type(paper))

# print("\nKeys inside this paper:")
# print(paper.keys())


#Now we need to inspect one paper's contents, because the list of 118 IDs alone doesn't tell us where the questions, answers, figures, and tables are.

# import json

# file_path = "data/test-A/SPIQA_testA.json"

# with open(file_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Take the first paper
# paper_id = list(data.keys())[0]
# paper = data[paper_id]

# print("Paper ID:")
# print(paper_id)

# print("\nType of paper data:")
# print(type(paper))

# print("\nKeys inside this paper:")
# print(paper.keys())


#Next: inspect all_figures and qa

import json

file_path = "data/test-A/SPIQA_testA.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Get first paper
paper_id = list(data.keys())[0]
paper = data[paper_id]

print("Paper ID:")
print(paper["paper_id"])

# =========================
# FIGURES / TABLES
# =========================

print("\n===== ALL FIGURES =====")

figures = paper["all_figures"]

print("Type:", type(figures))
print("Number of figures/tables:", len(figures))

print("\nFigure/Table keys:")
print(figures.keys())

# Get first figure/table
first_key = next(iter(figures))
first_figure = figures[first_key]

print("\nFirst figure/table key:")
print(first_key)

print("\nFirst figure/table content:")
print(first_figure)

# =========================
# QA
# =========================

print("\n===== QA =====")

qa = paper["qa"]

print("Type:", type(qa))
print("Number of QA entries:", len(qa))

print("\nQA keys/items:")

if isinstance(qa, dict):
    print(qa.keys())
else:
    print(qa[0])