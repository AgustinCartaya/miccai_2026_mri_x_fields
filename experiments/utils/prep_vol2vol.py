
import os
import subprocess





def apply_vol2vol(
    fixed_img_path_name,
    moving_img_path_name,
    output_img_path_name,
    use_regheader=True,
    reg_path_name=None,
    verify=False,
    verbose=False,
    interp="trilinear",
    nearest=False,
    inverse=False,
    threads=1
):
    """
    Apply FreeSurfer mri_vol2vol to resample an image into a target space.

    Args:
        fixed_img_path_name (str): Target image defining output space.
        moving_img_path_name (str): Input volume to transform.
        output_img_path_name (str): Output resampled volume.
        reg_path_name (str): Registration file (.dat).
        use_regheader (bool): Use header-based alignment.
        verify (bool): Skip if output exists.
        verbose (bool): Print progress.
        interp (str): Interpolation method.
        nearest (bool): Use nearest-neighbor interpolation.
        inverse (bool): Apply inverse transform.
        threads (int): Unused (kept for API compatibility).

    Returns:
        bool or None
    """

    if verify and os.path.exists(output_img_path_name):
        print(
            f"vol2vol already done for: {moving_img_path_name}\n"
            f"file in: {output_img_path_name}"
        )
        return True

    command = [
        "mri_vol2vol",
        "--mov", moving_img_path_name,
        "--targ", fixed_img_path_name,
        "--o", output_img_path_name,
    ]

    # --- registration (mandatory) ---
    if use_regheader:
        command.append("--regheader")
    elif reg_path_name is not None:
        command.extend(["--reg", reg_path_name])
    else:
        raise ValueError(
            "You must provide either reg_path_name or set use_regheader=True"
        )

    # --- interpolation ---
    if nearest or interp == "nearest":
        command.append("--nearest")
    else:
        command.extend(["--interp", interp])

    # --- inverse transform ---
    if inverse:
        command.append("--inv")

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        if verbose:
            print(f"vol2vol done saved in: {output_img_path_name}")
    else:
        print("Error:", result.stderr)

    return None



# def apply_vol2vol(
#     fixed_img_path_name,
#     moving_img_path_name,
#     output_img_path_name,
#     # reg_path_name=None,
#     verify=False,
#     verbose=False,
#     interp="trilinear",
#     # inverse=False,
#     # use_regheader=False,
#     nearest=False,
#     license_file=None,
#     threads=1
# ):
#     """
#     Apply FreeSurfer mri_vol2vol to resample an image into a target space.

#     Args:
#         fixed_name (str): Fixed image (target space).
#         moving_name (str): Moving image (input volume to transform).
#         output_name (str): Output resampled volume.
#         reg_path_name (str): Registration file (.dat). If None, uses --regheader if enabled.
#         verify (bool): If True, skip if output already exists.
#         verbose (bool): If True, print progress info.
#         interp (str): Interpolation method ("trilinear", "nearest", "cubic").
#         inverse (bool): If True, apply inverse transform (--inv).
#         use_regheader (bool): If True, use header-based alignment (--regheader).
#         nearest (bool): Shortcut for nearest-neighbor interpolation (overrides interp).
#         threads (int): (not used by mri_vol2vol, kept for API consistency).

#     Returns:
#         bool or None: True if skipped due to verify, None otherwise.
#     """

#     if verify:
#         if os.path.exists(output_img_path_name):
#             print(f"vol2vol already done for: {moving_img_path_name}\nfile in: {output_img_path_name}")
#             return True

#     command = [
#         "mri_vol2vol",
#         "--mov", moving_img_path_name,
#         "--targ", fixed_img_path_name,
#         "--o", output_img_path_name,
#         "--license", license_file

#     ]

#     # Registration handling
#     # if use_regheader:
#     #     command.append("--regheader")
#     # elif reg_path_name is not None:
#     #     command.extend(["--reg", reg_path_name])
#     # else:
#     #     raise ValueError("You must provide either reg_path_name or set use_regheader=True")

#     # Interpolation
#     if nearest:
#         command.append("--nearest")
#     else:
#         command.extend(["--interp", interp])

#     # Inverse transform
#     # if inverse:
#     #     command.append("--inv")

#     result = subprocess.run(command, capture_output=True, text=True)

#     if result.returncode == 0:
#         if verbose:
#             print(f"vol2vol done saved in: {output_img_path_name}")
#     else:
#         print("Error:", result.stderr)

#     return None