def model_hyperlink(model_link, code_link, model_name):
    if model_link == "":
        return model_name
        # return f'<a target="_blank">{model_name}</a>'
        # return f'<a target="_blank" href="{link}" rel="noopener noreferrer">{model_name}</a>'
    else:
        model_url = f'<a target="_blank" href="{model_link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'
        if code_link == "":
            return model_url
        else:
            code_url = f'<a target="_blank" href="{code_link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">code</a>'
            return f"{model_url} ({code_url})"
    # return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a> | ' \
    #         f'<a target="_blank" href="https://www.google.com" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}_link2</a>'


def make_clickable_model(model_name):
    link = f"https://huggingface.co/{model_name}"
    return model_hyperlink(link, model_name)


def styled_error(error):
    return f"<p style='color: red; font-size: 20px; text-align: center;'>{error}</p>"


def styled_warning(warn):
    return f"<p style='color: orange; font-size: 20px; text-align: center;'>{warn}</p>"


def styled_message(message):
    return f"<p style='color: green; font-size: 20px; text-align: center;'>{message}</p>"


def has_no_nan_values(df, columns):
    return df[columns].notna().all(axis=1)


def has_nan_values(df, columns):
    return df[columns].isna().any(axis=1)
