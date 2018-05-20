
def make_arff(relation_name, attrs, data):
    """
    :param relation_name: the relation name of arff
    :param attrs: attribute tuples of (name, data type)
    :param data: a list of data according to attrs
    :return: the arff string
    """
    assert len(attrs) == len(data[0]), "The lengths of attributes and data are not matched."

    relation_raw = "@relation " + relation_name + "\n\n"
    attrs_raw = ""
    for attr in attrs:
        attrs_raw += "@attribute {} {}\n".format(*attr)

    data_raw = "\n@data\n"
    for d in data:
        d = [str(n) if isinstance(n, int) else "\""+str(n).replace("\"", "'")+"\"" for n in d]
        data_raw += ",".join(d) + "\n"

    return relation_raw + attrs_raw + data_raw
