"""Small formatting and model-inspection helpers used by training."""


def format_count(value):
    """Format a large count using K/M/B suffixes for console output."""
    value = float(value)
    absolute_value = abs(value)
    if absolute_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if absolute_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if absolute_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.0f}"


def get_model_parameter_metrics(model):
    """Return unique parameter counts and storage size for a model."""
    parameters = list(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in parameters)
    trainable = sum(
        parameter.numel() for _, parameter in parameters if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for _, parameter in parameters
    )
    return {
        "model/parameters_total": total,
        "model/parameters_trainable": trainable,
        "model/parameters_non_trainable": total - trainable,
        "model/parameter_memory_bytes": parameter_bytes,
        "model/parameter_memory_mb": parameter_bytes / 1024**2,
    }
