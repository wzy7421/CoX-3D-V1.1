def shap_attribution(model, features):
    """
    Placeholder SHAP wrapper. Plug your trained semantic-to-score model here.
    """
    try:
        import shap
    except Exception as e:
        raise RuntimeError("shap is not installed. Install shap to use this module.") from e
    explainer = shap.Explainer(model)
    return explainer(features)
