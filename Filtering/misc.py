from utils import embedding_vector, cos_sim, embedding_vectors
def cos_sim_chart():
    words = [" science", " scientist", " matrix", " wizard", " magic", " fantasy"]
    lens = [len(embedding_vectors(w)) for w in words]
    for l in lens:
        assert l == 1
    tokens = [embedding_vector(w) for w in words]
    output = []
    for t1 in tokens:
        row = []
        for t2 in tokens:
            row.append(cos_sim(t1, t2))
        output.append(row)
    return output


def print_latex_table(arr):
    for row in arr:
        row_str = ""
        for num in row:
            if "{:.2f}&".format(num) == "1.00&":
                row_str += "\\textbf{1.00}&"
            else:
                row_str += "{:.2f}&".format(num)
        print(row_str[:-1] + "\\\\")
        print("\\hline")

print_latex_table(cos_sim_chart())