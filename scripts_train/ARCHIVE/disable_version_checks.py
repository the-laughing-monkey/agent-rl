import transformers.utils.versions as tv

def no_op_require_version(package, *args, **kwargs):
    # Optionally, print a warning message
    print(f"Warning: Skipping version check for package '{package}'.")

# Override the require_version function with our no-op version.
tv.require_version = no_op_require_version