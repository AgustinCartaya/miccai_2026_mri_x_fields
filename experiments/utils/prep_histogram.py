import numpy as np

def compute_hist(
    img,
    mask=None,
    quantiles=np.linspace(0, 100, 1001),
    lower_clip=0,
    upper_clip=100,
):
    if mask is None:
        mask = np.ones_like(img, dtype=bool)

    voxels = img[mask > 0]

    if len(voxels) == 0:
        return None

    low = np.percentile(voxels, lower_clip)
    high = np.percentile(voxels, upper_clip)

    voxels = voxels[(voxels >= low) & (voxels <= high)]

    return np.percentile(voxels, quantiles)



def compute_mean_hist(
    img_list,
    mask_list=None,
    quantiles=np.linspace(0, 100, 1001),
    lower_clip=0,
    upper_clip=100,
):
    """
    Compute a mean reference distribution from a dataset.

    Parameters
    ----------
    img_list : list[np.ndarray]
    mask_list : list[np.ndarray]
    quantiles : np.ndarray
    lower_clip : float
    upper_clip : float

    Returns
    -------
    reference_quantiles : np.ndarray
    """

    all_quantiles = []

    for i, img in enumerate(img_list):

        if mask_list is not None:
            mask = mask_list[i]
        else:
            mask = None


        q  = compute_hist(
            img,
            mask=mask,
            quantiles=quantiles,
            lower_clip=lower_clip,
            upper_clip=upper_clip,
        )
        all_quantiles.append(q)

    reference_quantiles = np.mean(
        np.stack(all_quantiles, axis=0),
        axis=0
    )

    return reference_quantiles



def match_hist_with_reference(
    image,
    reference_quantiles,
    mask=None,
    quantiles=np.linspace(0, 100, 1001),
):
    """
    Match image intensities to a reference distribution.

    Parameters
    ----------
    image : np.ndarray
    mask : np.ndarray
    reference_quantiles : np.ndarray

    Returns
    -------
    matched_image : np.ndarray
    """

    if mask is None:
        mask = np.ones_like(image, dtype=bool)

    matched = image.copy()

    voxels = image[mask > 0]

    source_quantiles = np.percentile(
        voxels,
        quantiles
    )

    matched[mask > 0] = np.interp(
        voxels,
        source_quantiles,
        reference_quantiles
    )

    return matched



def match_hist_with_image(
    image,
    reference_image,
    mask=None,
    reference_mask=None,
    quantiles=np.linspace(0, 100, 1001),
):
    """
    Match image intensities to a reference distribution.

    Parameters
    ----------
    image : np.ndarray
    mask : np.ndarray
    reference_image : np.ndarray
    reference_mask : np.ndarray

    Returns
    -------
    matched_image : np.ndarray
    """

    q_reference = compute_hist(
        reference_image,
        mask=reference_mask,
        quantiles=quantiles
    )

    matched = match_hist_with_reference(
        image=image,
        reference_quantiles=q_reference,
        mask=mask,
        quantiles=quantiles
    )
    return matched











def compute_mean_hist_tissuewise(
    img_list,
    seg_list,
    quantiles=np.linspace(0, 100, 1001),
    lower_clip=0,
    upper_clip=100,
    tissues_idx = None
):
    """
    Compute mean quantiles for each tissue.
    """

    reference = []
    if tissues_idx is None:
        tissues_idx = np.unique(np.concatenate(seg_list))


    for tissue in tissues_idx:

        all_subject_quantiles = []

        for img, seg in zip(img_list, seg_list):

            mask = seg == tissue
            voxels = img[mask]

            low = np.percentile(voxels, lower_clip)
            high = np.percentile(voxels, upper_clip)

            voxels = voxels[
                (voxels >= low) &
                (voxels <= high)
            ]

            all_subject_quantiles.append(
                np.percentile(voxels, quantiles)
            )

        reference.append(np.mean(
            np.stack(all_subject_quantiles),
            axis=0
        ))

    return np.array(reference)

def build_global_landmark_mapping(
    img,
    seg,
    reference_quantiles,
    quantiles=np.linspace(0, 100, 1001),
    tissues_idx=None,
    lower_clip=0,
    upper_clip=100,
    min_voxels=10,
):

    src_landmarks = []
    tgt_landmarks = []

    if tissues_idx is None:
        tissues_idx = np.unique(seg)

    for i, tissue in enumerate(tissues_idx):

        voxels = img[seg == tissue]

        if len(voxels) < min_voxels:
            continue

        low = np.percentile(voxels, lower_clip)
        high = np.percentile(voxels, upper_clip)

        voxels = voxels[
            (voxels >= low) &
            (voxels <= high)
        ]

        src_q = np.percentile(
            voxels,
            quantiles
        )

        tgt_q = reference_quantiles[i]

        src_landmarks.append(src_q)
        tgt_landmarks.append(tgt_q)

    if len(src_landmarks) == 0:
        raise ValueError("No valid tissues found.")

    src_landmarks = np.concatenate(src_landmarks)
    tgt_landmarks = np.concatenate(tgt_landmarks)

    idx = np.argsort(src_landmarks)

    src_landmarks = src_landmarks[idx]
    tgt_landmarks = tgt_landmarks[idx]

    # Average duplicated source landmarks
    unique_src = np.unique(src_landmarks)

    unique_tgt = np.array([
        tgt_landmarks[src_landmarks == s].mean()
        for s in unique_src
    ])

    assert len(unique_src) == len(unique_tgt)
    assert np.all(np.diff(unique_src) > 0)

    return unique_src, unique_tgt


def match_histogram_quantiles_tissuewise(
    image,
    seg,
    reference_quantiles,
    brain_mask=None,
    quantiles=np.linspace(0, 100, 1001),
    tissues_idx=None,
    lower_clip=0,
    upper_clip=100,
):

    if brain_mask is None:
        brain_mask = np.ones_like(
            image,
            dtype=bool
        )

    matched = image.copy()

    src_landmarks, tgt_landmarks = (
        build_global_landmark_mapping(
            image,
            seg,
            reference_quantiles,
            quantiles=quantiles,
            tissues_idx=tissues_idx,
            lower_clip=lower_clip,
            upper_clip=upper_clip,
        )
    )

    matched[brain_mask] = np.interp(
        image[brain_mask],
        src_landmarks,
        tgt_landmarks,
    )

    return matched