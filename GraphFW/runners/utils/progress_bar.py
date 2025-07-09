import sys

def progress_bar(
    prior_vars: dict = None,
    posterior_vars: dict = None,
    prefix='',
    postfix='',
    style='bar',
    bar_width=10
):
    """
    Display a live training progress bar in the console.
    Args:
        prior_vars (dict, optional): Variables to display before the progress bar.
        posterior_vars (dict, optional): Variables to display after the progress bar.
        prefix (str, optional): Prefix string to display before the progress bar.
        postfix (str, optional): Postfix string to display after the progress bar.
        style (str, optional): Style of the progress bar ('bar', 'arrow', 'dots').
        bar_width (int, optional): Width of the progress bar.
    """
    # Extract progress info from prior_vars if present
    s = ''

    epoch = prior_vars.get('epoch') if prior_vars else None
    total_epochs = prior_vars.get('total_epochs') if prior_vars else None
    iteration = prior_vars.get('iteration') if prior_vars else None
    total_iterations = prior_vars.get('total_iterations') if prior_vars else None

    progress = 0
    if iteration is not None and total_iterations:
        progress = iteration / total_iterations
    elif epoch is not None and total_epochs:
        progress = epoch / total_epochs
    filled_len = int(progress * bar_width)
    if style == 'bar':
        progress_vis = "█" * filled_len + '-' * (bar_width - filled_len)
    elif style == 'arrow':
        progress_vis = ">" * filled_len + '-' * (bar_width - filled_len)
    elif style == 'dots':
        progress_vis = "•" * filled_len + '-' * (bar_width - filled_len)
    else:
        progress_vis = "█" * filled_len + '-' * (bar_width - filled_len)

    if prefix:
        s += prefix + ' | '
    # Print prior_vars before progress bar
    if prior_vars:
        if epoch is not None and total_epochs:
            width = len(str(total_epochs))
            s += f"Epoch {epoch:0{width}d}/{total_epochs} | "
        if iteration is not None and total_iterations:
            width = len(str(total_iterations))
            s += f"Iteration {iteration:0{width}d}/{total_iterations} | "

        for key, value in prior_vars.items():
            if key in ['epoch', 'total_epochs', 'iteration', 'total_iterations']:
                continue
            if isinstance(value, float):
                s += f"{key}: {value:.4f} | "
            else:
                s += f"{key}: {value} | "
    s += f"[{progress_vis}] | "
    if postfix:
        s += postfix + ' | '
    # Print posterior_vars after progress bar
    if posterior_vars:
        for key, value in posterior_vars.items():
            if isinstance(value, float):
                s += f"{key}: {value:.4f} | "
            else:
                s += f"{key}: {value} | "
    s = s.rstrip(' | ')
    sys.stdout.write('\r' + s)
    sys.stdout.flush()
