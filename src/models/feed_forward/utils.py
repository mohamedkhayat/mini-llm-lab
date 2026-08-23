def equalize_params(hidden_dim, gated, do_equalize=False):
    if not do_equalize or not gated:
        return hidden_dim
    return int(2/3 * hidden_dim)