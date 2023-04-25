def load_data(path, load_func, columnnames_to_lower=True):
    data = load_func(path)

    if columnnames_to_lower:
        if isinstance(data, dict):
            for table in data.values():
                table.columns = table.columns.str.lower()
        else:
            data.columns = data.columns.str.lower()
    return data
